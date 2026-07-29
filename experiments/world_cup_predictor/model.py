"""Poisson goal model for international football.

Goals for each side are independent Poisson counts with a log link:

    log(lambda_ij) = mu + attack_i - defence_j + home_adv * (not neutral)

Dataset: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Machinery only. No printing, no plotting, nothing that runs on import. analysis.ipynb
imports from here; this module does not know the notebook exists.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

DATA = Path(__file__).parent / "data" / "results.csv"
FORMER_NAMES = Path(__file__).parent / "data" / "former_names.csv"
SHOOTOUTS = Path(__file__).parent / "data" / "shootouts.csv"
ALLOCATION = Path(__file__).parent / "data" / "third_place_allocation.csv"

# Knockout rounds in order, and how many ties each holds. A 48 team World Cup starts its
# knockout at the round of 32. The third place playoff is left out: it hangs off the semi
# final losers, feeds nothing, and would break the tree.
ROUNDS = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "Final": 1}

GROUP_MATCHES = 72     # twelve groups of four, three matches each
GROUP_SIZE = 4
THIRD_PLACE_SLOTS = 8  # best third placed teams joining the twelve winners and runners up

# One team per group, enough to label all twelve. Ten of these fall out of the data: read the
# actual round of 32 against the flagged row of the allocation table and 1A played 3E, where
# 1A was Mexico, so Mexico's group is A. C and H never appear in a third placed slot, so they
# come from the published group tables.
GROUP_ANCHORS = {"A": "Mexico", "B": "Switzerland", "C": "Brazil", "D": "United States",
                 "E": "Germany", "F": "Sweden", "G": "Belgium", "H": "Spain",
                 "I": "France", "J": "Algeria", "K": "Colombia", "L": "England"}

# winning a round of 32 tie puts you in the round of 16, and so on up to the trophy
NEXT_ROUND = dict(zip(ROUNDS, list(ROUNDS)[1:] + ["Winner"]))

# Tournament opening fixtures, passed to fit() as the cutoff. Training is strictly before
# the cutoff, so a tournament never contributes to the fit that predicts it.
CUTOFF_2022 = "2022-11-20"   # 2022 WC ran 2022-11-20 to 2022-12-18   (64 matches)
CUTOFF_2026 = "2026-06-11"   # 2026 WC ran 2026-06-11 to 2026-07-19  (104 matches)
FINAL_2026 = "2026-07-19"    # the final itself, so a fit at this cutoff has seen the other 103

# Every tuning decision was made with the cutoff at CUTOFF_2022, scored on the window
# between the two tournaments. The later cutoffs exist only to make final predictions with
# the settings already frozen, never to choose anything.

# Scoreline grid runs 0..MAX_GOALS a side. 10 was not enough: the worst mismatches predict
# goal rates above 15, where a grid that stops at 10 throws away most of the distribution
# and returns three probabilities summing to 0.08. The grid is normalised as well, so the
# result is a valid distribution whatever the rates do.
MAX_GOALS = 25

OUTCOMES = ["home win", "draw", "away win"]

# Match type, used only to stratify the scores. It used to weight the fit as well, on the
# reasoning that friendlies carry less signal, and sweeping that showed the assumption
# backwards: the more weight friendlies got the better the fit scored, on World Cup matches
# as much as overall, while the World Cup weight itself did nothing measurable. Qualifying is
# regional, so friendlies are where confederations meet and carry most of the evidence tying
# one continent's scale to another's. Weighting the objective by type was tried too and
# changed no tuning decision, since 64 World Cup matches cannot move a 3500 match average.
TAGS = ["World Cup", "FIFA", "Friendly", "OTHER"]


@dataclass
class PoissonFit:
    """Fitted coefficients. Everything needed to predict any fixture."""
    attack: pd.Series      # per team, higher scores more
    defence: pd.Series     # per team, higher concedes fewer
    home_adv: float        # log scale, applied only off neutral ground
    intercept: float       # log baseline goal rate
    rho: float = 0.0       # Dixon Coles low score correction, 0 disables it

    def expected_goals(self, home_team, away_team, neutral=True):
        """(lambda_home, lambda_away) for one fixture, as floats.

        Convenience wrapper over goal_rates for a single pairing, so there is only one
        implementation of the rate calculation to keep correct.
        """
        fixture = pd.DataFrame({"home_team": [home_team], "away_team": [away_team],
                                "neutral": [neutral]})
        home, away = goal_rates(self, fixture)
        return home[0], away[0]


def load(path=DATA, since="2000-01-01"):
    """Read results.csv from `since` onward, renamed, filtered to FIFA members, tagged.

    Returns unweighted matches. The fit's only weighting is time decay, applied in fit().

    No cutoff here. Date filtering for training lives in split().

    `since` is deliberately generous. Undecayed, the older matches make predictions worse;
    with decay on they are harmless and marginally useful, so the half life does this job
    smoothly and there is no reason for a hard boundary to do it badly. Reaching back to
    1990 or 1970 changes nothing measurable and doubles the fit time.
    """
    results = pd.read_csv(path, parse_dates=["date"])

    columns = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    games = results.loc[results["date"] >= since, columns].copy()

    # The download runs a few days ahead of itself: the newest fixtures ship with no score,
    # so drop them once, here. A scoreless row is meaningless to every caller downstream.
    games = games.dropna(subset=["home_score", "away_score"])
    games[["home_score", "away_score"]] = games[["home_score", "away_score"]].astype(int)

    # Fold historic names into the current one so a team keeps a single coefficient across a
    # rename. A no-op from 2000 on, since no former name appears in the data after then, but
    # it matters as soon as `since` reaches back past Serbia and Montenegro or the USSR.
    renames = pd.read_csv(FORMER_NAMES)
    games[["home_team", "away_team"]] = games[["home_team", "away_team"]].replace(
        dict(zip(renames.former, renames.current)))

    # collapse tournament names into World Cup / FIFA (qualifiers, Series) / Friendly / OTHER
    conditions = [
        games["tournament"] == "FIFA World Cup",
        games["tournament"].isin(["FIFA World Cup qualification", "FIFA Series"]),
        games["tournament"] == "Friendly",
    ]
    games["WC_tag"] = np.select(conditions, TAGS[:3], default="OTHER")

    # Keep FIFA members only. Playing a qualifier or a World Cup is what membership means in
    # practice, so the roster derives from the data rather than a pasted list, and it comes
    # out at exactly the 211 member associations. Drops regions, territories, unrecognised
    # states and diaspora sides, none of which can reach a World Cup.
    competitive = games.WC_tag.isin(["FIFA", "World Cup"])
    members = pd.unique(games.loc[competitive, ["home_team", "away_team"]].values.ravel())
    games = games[games.home_team.isin(members) & games.away_team.isin(members)]

    return games.drop(columns=["tournament"])


def to_long(games):
    """One row per team per match: goals scored, opponent conceding, venue.

    Expects a `weight` column, which fit() attaches. load() does not, since the only
    weighting in the model is the time decay fit() computes from the cutoff.

    The regression needs who scored, against whom, how many. Each match becomes two rows,
    the home side attacking the away defence with the home term, then the reverse without.
    """
    scoring_at_home = pd.DataFrame({
        "attacker": games.home_team,
        "defender": games.away_team,
        "goals": games.home_score,
        "at_home": (~games.neutral).astype(float),
        "weight": games.weight,
    })
    scoring_away = pd.DataFrame({
        "attacker": games.away_team,
        "defender": games.home_team,
        "goals": games.away_score,
        "at_home": 0.0,
        "weight": games.weight,
    })
    return pd.concat([scoring_at_home, scoring_away], ignore_index=True)


def goal_rates(params, games):
    """(lambda_home, lambda_away) arrays for a whole fixture list at once.

    The one place the linear predictor is evaluated. A team absent from the fit falls back
    to coefficient 0, the ridge prior, so it predicts as average. Since the FIFA member
    filter went in nothing in the holdout hits that, but a member with no matches before
    the cutoff still could.
    """
    attack_home = games.home_team.map(params.attack).fillna(0.0).to_numpy()
    attack_away = games.away_team.map(params.attack).fillna(0.0).to_numpy()
    defence_home = games.home_team.map(params.defence).fillna(0.0).to_numpy()
    defence_away = games.away_team.map(params.defence).fillna(0.0).to_numpy()
    bonus = np.where(games.neutral.to_numpy(), 0.0, params.home_adv)
    return (np.exp(params.intercept + attack_home - defence_away + bonus),
            np.exp(params.intercept + attack_away - defence_home))


def tau(home_goals, away_goals, rate_home, rate_away, rho):
    """Dixon Coles correction factor for a scoreline. 1 everywhere outside the four cells.

    Negative rho moves probability onto 0-0 and 1-1 and off 1-0 and 0-1, which is to say
    onto draws. The four factors are built so the total probability is unchanged, so no
    renormalisation is needed afterwards.
    """
    factor = np.ones(np.broadcast(home_goals, away_goals, rate_home, rate_away).shape)
    factor = np.where((home_goals == 0) & (away_goals == 0), 1 - rate_home * rate_away * rho, factor)
    factor = np.where((home_goals == 0) & (away_goals == 1), 1 + rate_home * rho, factor)
    factor = np.where((home_goals == 1) & (away_goals == 0), 1 + rate_away * rho, factor)
    factor = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, factor)
    return factor


def fit_rho(params, games, weights):
    """Maximum likelihood rho with the goal rates held fixed. Returns a float.

    Only the four corrected cells depend on rho, and the Poisson part of the likelihood
    does not, so the objective is just the weighted log of the correction factors. Bounds
    come from requiring every factor to stay positive on the training data.
    """
    rate_home, rate_away = goal_rates(params, games)
    home_goals = games.home_score.to_numpy()
    away_goals = games.away_score.to_numpy()

    margin = 1e-6
    low = -1.0 / max(rate_home.max(), rate_away.max()) + margin
    high = min(1.0, 1.0 / (rate_home * rate_away).max()) - margin

    def negative_log_likelihood(candidate):
        factor = tau(home_goals, away_goals, rate_home, rate_away, candidate)
        return -np.sum(weights * np.log(factor))

    return minimize_scalar(negative_log_likelihood, bounds=(low, high), method="bounded").x


def split(df, cutoff, end=None):
    """(train, test): matches before `cutoff`, and matches in [cutoff, end).

    The only date filtering in the codebase. fit() routes through it and so does the
    notebook, so cell execution order cannot leak post cutoff results into a fit.

    `end` is what seals the final test: pass CUTOFF_2026 and the 2026 World Cup falls
    outside both halves, unseen by any tuning decision.
    """
    train = df[df.date < cutoff]
    test = df[df.date >= cutoff]
    return train, test if end is None else test[test.date < end]


def fit(df, cutoff, half_life_days=None, alpha=1e-6, dixon_coles=False):
    """Weighted Poisson regression on everything before `cutoff`. Returns a PoissonFit.

    One column per team for attack and one per team for defence. Defence columns are
    negated so both sets of coefficients run higher is better.

    `alpha` is a ridge penalty pulling coefficients towards average. Swept from 1e-6 to
    0.1: flat below 1e-5 and worse above it, so shrinkage is not a binding constraint here.
    Every member has enough matches that the prior only adds bias, and the FIFA filter
    already removed the two cap teams it would otherwise be protecting against.

    `half_life_days` weights each match by 0.5 ** (age / half_life), age measured back from
    the cutoff, so coefficients track current strength rather than a decade average. None
    disables it and every match counts the same. This is the only weighting in the fit.

    `dixon_coles` adds a second stage estimating rho on the same training matches with the
    goal rates held fixed. Not the joint maximum likelihood, which would mean replacing the
    GLM entirely, but the bias from holding the rates fixed is second order.
    """
    train, _ = split(df, cutoff)
    if train.empty:
        raise ValueError(f"no matches before {cutoff}")

    age_days = (pd.Timestamp(cutoff) - train.date).dt.days
    decay = 1.0 if half_life_days is None else 0.5 ** (age_days / half_life_days)
    train = train.assign(weight=decay)

    long = to_long(train)

    attack_cols = pd.get_dummies(long.attacker, prefix="attack").astype(float)
    defence_cols = -pd.get_dummies(long.defender, prefix="defence").astype(float)
    design = pd.concat([attack_cols, defence_cols, long[["at_home"]]], axis=1)

    regressor = PoissonRegressor(alpha=alpha, max_iter=1000)
    regressor.fit(design, long.goals, sample_weight=long.weight)

    coefficients = pd.Series(regressor.coef_, index=design.columns)
    params = PoissonFit(
        attack=coefficients.filter(like="attack_").rename(lambda c: c.removeprefix("attack_")),
        defence=coefficients.filter(like="defence_").rename(lambda c: c.removeprefix("defence_")),
        home_adv=coefficients["at_home"],
        intercept=regressor.intercept_,
    )
    if dixon_coles:
        params.rho = fit_rho(params, train, train.weight.to_numpy())
    return params


def score_grid(rate_home, rate_away, rho=0.0):
    """P(every scoreline) as a (MAX_GOALS+1, MAX_GOALS+1) array, grid[i, j] = P(i-j).

    Everything the model knows about a fixture. The 1X2 probabilities are a summary of
    this, and a tournament simulation needs the scorelines themselves, since group tables
    are decided on goal difference and goals scored.
    """
    home_goals = poisson.pmf(np.arange(MAX_GOALS + 1), rate_home)
    away_goals = poisson.pmf(np.arange(MAX_GOALS + 1), rate_away)
    grid = np.outer(home_goals, away_goals)

    if rho:
        corner = np.indices((2, 2))
        grid[:2, :2] *= tau(corner[0], corner[1], rate_home, rate_away, rho)
    return grid / grid.sum()  # the tail beyond MAX_GOALS, tiny but not always zero


def outcome_probs(rate_home, rate_away, rho=0.0):
    """1X2 probabilities for one fixture, from a pair of goal rates."""
    grid = score_grid(rate_home, rate_away, rho)
    home_win = np.tril(grid, -1).sum()  # below the diagonal, home scored more
    draw = np.trace(grid)               # on the diagonal, level
    away_win = np.triu(grid, 1).sum()   # above the diagonal, away scored more
    return home_win, draw, away_win


def match_probs(params, home, away, neutral=True):
    """1X2 probabilities for a single pairing. Neutral ground by default, as a tournament is."""
    return outcome_probs(*params.expected_goals(home, away, neutral), params.rho)


def predict(params, games):
    """Outcome probabilities for a set of fixtures, as an (n, 3) array."""
    rate_home, rate_away = goal_rates(params, games)
    return np.array([outcome_probs(home, away, params.rho)
                     for home, away in zip(rate_home, rate_away)])


def likely_scores(params, home, away, neutral=True, n=6):
    """The `n` most probable scorelines for one fixture, as a Series indexed by "2-1"."""
    grid = score_grid(*params.expected_goals(home, away, neutral), params.rho)
    labels = [f"{i}-{j}" for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    return pd.Series(grid.ravel(), index=labels).nlargest(n).rename_axis("scoreline")


def advance_probs(params, home, away, neutral=True):
    """(P(home advances), P(away advances)) in a knockout tie.

    A knockout tie cannot end level. The dataset records scores after extra time, so a
    draw here means penalties, and shootouts are close to coin flips: team strength barely
    predicts them and the shooting order is itself decided by a coin toss. So the draw
    probability splits evenly rather than being simulated.
    """
    home_win, draw, away_win = match_probs(params, home, away, neutral)
    return home_win + 0.5 * draw, away_win + 0.5 * draw


def knockout_ties(games, group_matches=72):
    """The knockout matches of a tournament, in round order, with the tie winner attached.

    `group_matches` is how many fixtures the group stage holds: 72 for the 48 team format,
    twelve groups of four playing three each. Everything after that in date order is the
    knockout. Anchoring to the start of the knockout rather than counting back from the end
    means a tournament whose last fixtures are missing or newly added shifts nothing.

    A level score means the tie went to penalties, so the winner comes from shootouts.csv
    rather than the scoreline. Only the rounds that feed the tree are returned, so the
    final is absent: it is the one fixture whose participants are entirely determined by
    the rounds below it.
    """
    played = games.sort_values("date").iloc[group_matches:]
    feeding_rounds = [name for name in ROUNDS if name != "Final"]
    ties = played.iloc[:sum(ROUNDS[name] for name in feeding_rounds)].copy()

    shootouts = pd.read_csv(SHOOTOUTS, parse_dates=["date"])
    decided = {(d, h, a): w for d, h, a, w in
               zip(shootouts.date, shootouts.home_team, shootouts.away_team, shootouts.winner)}

    winners = []
    for date, home, away, home_score, away_score in zip(
            ties.date, ties.home_team, ties.away_team, ties.home_score, ties.away_score):
        if home_score > away_score:
            winners.append(home)
        elif away_score > home_score:
            winners.append(away)
        else:
            winners.append(decided[(date, home, away)])
    ties["winner"] = winners
    ties["round"] = [name for name in feeding_rounds for _ in range(ROUNDS[name])]

    opening = ties[ties["round"] == "R32"]
    distinct = pd.concat([opening.home_team, opening.away_team]).nunique()
    if distinct != 2 * ROUNDS["R32"]:
        raise ValueError(f"round of 32 has {distinct} distinct teams, expected {2 * ROUNDS['R32']}")

    return ties.reset_index(drop=True)


def group_letters(games):
    """{team: group letter} for a tournament's group stage.

    The groups themselves need no external source: within the group stage every team plays
    exactly the three others in its group, so the match graph falls into twelve components
    of four. Only the labelling needs anchors, and letters matter because the third place
    allocation table is keyed by them.
    """
    graph = nx.Graph()
    graph.add_edges_from(zip(*games.sort_values("date").iloc[:GROUP_MATCHES]
                             [["home_team", "away_team"]].values.T))
    components = [set(c) for c in nx.connected_components(graph)]
    if len(components) != len(GROUP_ANCHORS) or any(len(c) != GROUP_SIZE for c in components):
        raise ValueError(f"group stage is not {len(GROUP_ANCHORS)} groups of {GROUP_SIZE}")

    letters = {}
    for letter, anchor in GROUP_ANCHORS.items():
        component = next(c for c in components if anchor in c)
        letters.update({team: letter for team in component})
    if len(set(letters.values())) != len(GROUP_ANCHORS):
        raise ValueError("two anchors landed in the same group")
    return letters


def group_table(games, letters):
    """Group standings: points, then goal difference, then goals scored, with a position.

    FIFA breaks any remaining tie on head to head, then fair play, then lots. The first is
    fiddly and the last two are unmodellable, so ties beyond goals scored fall to the order
    pandas happens to produce. On the real tournament nothing gets that far.
    """
    rows = []
    for home, away, home_goals, away_goals in zip(games.home_team, games.away_team,
                                                  games.home_score, games.away_score):
        for team, scored, conceded in ((home, home_goals, away_goals), (away, away_goals, home_goals)):
            rows.append({"team": team, "group": letters[team], "for": scored, "against": conceded,
                         "points": 3 if scored > conceded else 1 if scored == conceded else 0})

    table = pd.DataFrame(rows).groupby(["group", "team"], as_index=False).sum()
    table["gd"] = table["for"] - table["against"]
    table = table.sort_values(["group", "points", "gd", "for"], ascending=[True, False, False, False])
    table["position"] = table.groupby("group").cumcount() + 1
    return table.reset_index(drop=True).rename_axis(columns="group stage")


def qualifying_thirds(table, places=THIRD_PLACE_SLOTS):
    """The `places` best third placed teams, ranked on the same keys as the group tables."""
    thirds = table[table.position == 3]
    return thirds.sort_values(["points", "gd", "for"], ascending=False).head(places)


def third_place_allocation(path=ALLOCATION):
    """{"BDEFIJKL": {"1A": "3E", ...}}, FIFA's published slot table.

    Which third placed team a group winner faces depends on which eight groups produced the
    qualifying thirds, and there are 495 such combinations. The table is published rather
    than derivable, so it is cached beside the results rather than fetched.
    """
    table = pd.read_csv(path)
    slots = [column for column in table.columns if column.startswith("1")]
    return table.set_index("qualifying_thirds")[slots].to_dict(orient="index")


def slot_template(games, letters=None):
    """The round of 32 as position codes, one pair per fixture, in bracket node order.

    Returns [("2A", "2B"), ("1C", "2F"), ...] where "2F" means the runner up of group F. The
    eight ties involving a third placed team carry the winner's code and None, since which
    third they face is not fixed: it depends on which groups the qualifying thirds came from
    and has to be looked up per tournament.
    """
    letters = group_letters(games) if letters is None else letters
    table = group_table(games.sort_values("date").iloc[:GROUP_MATCHES], letters)
    qualified = set(qualifying_thirds(table).team)

    code = {row.team: f"{row.position}{row.group}" for row in table.itertuples()
            if row.position <= 2 or row.team in qualified}

    template = []
    for _, tie in knockout_ties(games).query("round == 'R32'").iterrows():
        pair = tuple(code[team] for team in (tie.home_team, tie.away_team))
        template.append(tuple(None if slot.startswith("3") else slot for slot in pair))
    return template


def bracket(ties):
    """The knockout tree as a networkx DiGraph, one node per fixture.

    Nodes are named "R32-0", "R16-3" and so on, and carry `round`. Round of 32 nodes also
    carry `teams`, the pairing that actually opened the tournament. Every edge points from
    a fixture to the one its winner feeds, so a topological sort gives the order to play
    them in and the single sink is the final.

    The tree above the round of 32 is recovered from who met whom in each later round,
    which is fixed tournament structure rather than a result: the shape was set before the
    tournament started.
    """
    graph = nx.DiGraph()
    opening = ties[ties["round"] == "R32"].reset_index(drop=True)
    for slot, tie in opening.iterrows():
        graph.add_node(f"R32-{slot}", round="R32", teams=(tie.home_team, tie.away_team))

    feeds = {tie.winner: f"R32-{slot}" for slot, tie in opening.iterrows()}
    for name in ["R16", "QF", "SF"]:
        played = ties[ties["round"] == name].reset_index(drop=True)
        next_feeds = {}
        for slot, tie in played.iterrows():
            node = f"{name}-{slot}"
            graph.add_node(node, round=name)
            graph.add_edge(feeds[tie.home_team], node)
            graph.add_edge(feeds[tie.away_team], node)
            next_feeds[tie.winner] = node
        feeds = next_feeds

    graph.add_node("Final-0", round="Final")
    for node in feeds.values():
        graph.add_edge(node, "Final-0")
    return graph


def play_bracket(graph, params, rng, grids=None, opening=None):
    """Play the bracket once. Returns {node: (home, away, home_goals, away_goals, winner)}.

    Fixtures are played in topological order, so a tie is only reached once both feeders
    have produced a winner. Scorelines are sampled from the full grid rather than from the
    1X2 summary, and a level result is settled by the coin flip that advance_probs assumes.

    `grids` is a cache of cumulative scoreline distributions keyed by pairing. Thousands of
    runs revisit the same pairings constantly, so building each grid once turns the cost of
    a run into one searchsorted per fixture.

    `opening` replaces the real round of 32 pairings with simulated ones, which is what a
    full tournament simulation needs once the group stage decides who is there.
    """
    if grids is None:
        grids = {}
    played = {}

    for node in nx.topological_sort(graph):
        if graph.nodes[node]["round"] == "R32":
            home, away = graph.nodes[node]["teams"] if opening is None else opening[node]
        else:
            home, away = (played[feeder][4] for feeder in sorted(graph.predecessors(node)))

        if (home, away) not in grids:
            grid = score_grid(*params.expected_goals(home, away, neutral=True), params.rho)
            grids[home, away] = np.cumsum(grid.ravel())
        drawn = np.searchsorted(grids[home, away], rng.random())
        home_goals, away_goals = divmod(drawn, MAX_GOALS + 1)

        if home_goals > away_goals:
            winner = home
        elif away_goals > home_goals:
            winner = away
        else:
            winner = home if rng.random() < 0.5 else away
        played[node] = (home, away, home_goals, away_goals, winner)
    return played


@dataclass
class BracketOdds:
    """What many simulated tournaments say."""
    reach: pd.DataFrame    # team x round, probability of reaching that round
    nodes: dict            # node -> Series, probability each team wins that tie


def simulate_bracket(graph, params, runs=10_000, seed=0):
    """Play the bracket `runs` times. Returns a BracketOdds.

    `reach` gives each team's probability of reaching each round, so "SF" is the chance of
    playing in the semi final and "Winner" the chance of lifting the trophy. Its columns sum
    to the number of places each round holds, 16, 8, 4, 2, 1, which is a free correctness
    check on the whole simulation.

    `nodes` is the same evidence cut by fixture rather than by team, which is what a bracket
    diagram needs: for each tie, who is likely to come out of it.
    """
    rng = np.random.default_rng(seed)
    reached = {name: Counter() for name in list(ROUNDS)[1:] + ["Winner"]}
    per_node = {node: Counter() for node in graph}
    grids = {}

    for _ in range(runs):
        played = play_bracket(graph, params, rng, grids)
        for node, (_, _, _, _, winner) in played.items():
            reached[NEXT_ROUND[graph.nodes[node]["round"]]][winner] += 1
            per_node[node][winner] += 1

    reach = pd.DataFrame(reached).fillna(0) / runs
    reach = reach.sort_values("Winner", ascending=False).rename_axis(
        index="team", columns="probability of reaching")
    nodes = {node: (pd.Series(counts).sort_values(ascending=False) / runs).rename_axis("team")
             for node, counts in per_node.items()}
    return BracketOdds(reach, nodes)


def _group_setup(games, params):
    """Everything the group stage simulation needs, computed once.

    Team order is by group then name, so each group owns four consecutive indices and the
    whole standings calculation becomes a reshape rather than a groupby.
    """
    letters = group_letters(games)
    order = sorted(letters, key=lambda team: (letters[team], team))
    index = {team: i for i, team in enumerate(order)}

    fixtures = games.sort_values("date").iloc[:GROUP_MATCHES]
    home = np.array([index[team] for team in fixtures.home_team])
    away = np.array([index[team] for team in fixtures.away_team])

    rate_home, rate_away = goal_rates(params, fixtures)
    grids = np.array([np.cumsum(score_grid(h, a, params.rho).ravel())
                      for h, a in zip(rate_home, rate_away)])
    return order, index, home, away, grids, sorted(set(letters.values()))


def simulate_tournament(games, params, graph, runs=10_000, seed=0):
    """Play whole tournaments, group stage included. Returns a BracketOdds.

    The bracket simulation on its own starts from the real round of 32, so every number it
    gives is conditional on the group stage having gone exactly as it did. This starts from
    nothing: 72 group matches, standings on points then goal difference then goals scored,
    the eight best third placed teams, FIFA's published table to decide which third each
    group winner meets, then the knockout.

    Group standings are computed with numpy rather than pandas, because a groupby per run
    over ten thousand runs costs hours where this costs seconds.
    """
    order, index, home, away, grids, letters = _group_setup(games, params)
    template = slot_template(games)
    allocation = third_place_allocation()
    opening_nodes = [node for node in graph if graph.nodes[node]["round"] == "R32"]

    rng = np.random.default_rng(seed)
    reached = {name: Counter() for name in ["R32"] + list(ROUNDS)[1:] + ["Winner"]}
    knockout_grids = {}
    width = MAX_GOALS + 1

    for _ in range(runs):
        drawn = (grids >= rng.random(len(grids))[:, None]).argmax(axis=1)
        home_goals, away_goals = drawn // width, drawn % width

        home_points = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
        away_points = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))
        size = len(order)
        points = np.bincount(home, home_points, size) + np.bincount(away, away_points, size)
        scored = np.bincount(home, home_goals, size) + np.bincount(away, away_goals, size)
        conceded = np.bincount(home, away_goals, size) + np.bincount(away, home_goals, size)

        # one sortable number per team, with a jitter so exact ties break at random rather
        # than by team name, which is what FIFA's remaining tiebreakers amount to for us
        key = points * 1e6 + (scored - conceded) * 1e3 + scored + rng.random(size) * 1e-3
        ranked = np.argsort(-key.reshape(len(letters), GROUP_SIZE), axis=1)
        slots = np.take_along_axis(np.arange(size).reshape(len(letters), GROUP_SIZE), ranked, axis=1)

        thirds = slots[:, 2]
        best = np.argsort(-key[thirds])[:THIRD_PLACE_SLOTS]
        qualified = sorted(letters[group] for group in best)
        faces = allocation["".join(qualified)]

        occupant = {f"{place + 1}{letters[group]}": order[slots[group, place]]
                    for group in range(len(letters)) for place in range(2)}
        occupant.update({f"3{letters[group]}": order[thirds[group]] for group in best})

        opening = {}
        for node, (first, second) in zip(opening_nodes, template):
            second = second if second is not None else faces[first]
            opening[node] = (occupant[first], occupant[second])

        for team in (side for pair in opening.values() for side in pair):
            reached["R32"][team] += 1
        played = play_bracket(graph, params, rng, knockout_grids, opening)
        for node, (_, _, _, _, winner) in played.items():
            reached[NEXT_ROUND[graph.nodes[node]["round"]]][winner] += 1

    reach = pd.DataFrame(reached).fillna(0) / runs
    reach = reach.sort_values("Winner", ascending=False).rename_axis(
        index="team", columns="probability of reaching")
    return BracketOdds(reach, {})


def actual_bracket(games, ties=None):
    """{node: (home, away, home_goals, away_goals, winner)} for how the knockout really went.

    The same shape play_bracket returns, so a figure can be drawn from either. The final is
    picked up from the last fixture of the tournament rather than from `ties`, which stops
    at the semi finals because everything above them is determined by what feeds in.
    """
    ties = knockout_ties(games) if ties is None else ties
    played = {}
    for name in [name for name in ROUNDS if name != "Final"]:
        for slot, (_, tie) in enumerate(ties[ties["round"] == name].iterrows()):
            played[f"{name}-{slot}"] = (tie.home_team, tie.away_team,
                                        int(tie.home_score), int(tie.away_score), tie.winner)

    final = games.sort_values("date").iloc[-1]
    winner = final.home_team if final.home_score > final.away_score else final.away_team
    played["Final-0"] = (final.home_team, final.away_team,
                         int(final.home_score), int(final.away_score), winner)
    return played


def bracket_positions(graph):
    """{node: (x, y)} laying the bracket out left to right, one column per round.

    Geometry only, no drawing. Opening ties are ordered by walking down from the final
    rather than by date, so siblings sit next to each other and the tree comes out planar.
    Ordering them by kickoff instead makes the edges cross and puts unrelated fixtures on
    top of one another, since the schedule does not follow the shape of the draw.

    Every later fixture sits level with the midpoint of the two it is fed by, which is what
    makes a bracket read as a bracket.
    """
    root = next(node for node in graph if graph.out_degree(node) == 0)

    order = []

    def walk(node):
        feeders = sorted(graph.predecessors(node))
        if not feeders:
            order.append(node)
        for feeder in feeders:
            walk(feeder)

    walk(root)

    columns = list(ROUNDS)
    positions = {node: (0, float(slot)) for slot, node in enumerate(order)}
    for node in nx.topological_sort(graph):
        if graph.nodes[node]["round"] != "R32":
            positions[node] = (columns.index(graph.nodes[node]["round"]),
                               float(np.mean([positions[f][1] for f in graph.predecessors(node)])))
    return positions


def strength(params):
    """Attack, defence and their sum per team, strongest first.

    Log scale and identified only up to the intercept, so read the values relatively.
    """
    table = pd.DataFrame({"attack": params.attack, "defence": params.defence})
    table["total"] = table.attack + table.defence
    return table.sort_values("total", ascending=False).rename_axis(index="team", columns="log scale")


def compare(before, after, n=15):
    """Rating and rank movement between two fits, for the top `n` teams of `after`.

    `moved` is positive when a team climbed. Ratings are log scale sums of attack and
    defence, so the change column is only meaningful relative to other teams in the table.
    """
    ratings = pd.DataFrame({"before": strength(before).total,
                            "after": strength(after).total}).dropna()
    ratings["change"] = ratings.after - ratings.before
    ratings["was"] = ratings.before.rank(ascending=False).astype(int)
    ratings["rank"] = ratings.after.rank(ascending=False).astype(int)
    ratings["moved"] = ratings.was - ratings["rank"]
    return (ratings.nsmallest(n, "rank")[["rank", "was", "moved", "before", "after", "change"]]
            .rename_axis(index="team", columns="rank, then rating on the log scale"))


def outcome_index(games):
    """0 if the home team won, 1 for a draw, 2 if the away team won."""
    return 1 - np.sign(games.home_score - games.away_score).to_numpy(dtype=int)


def brier(probs, actuals):
    """Multiclass Brier score. Proper, lower is better.

    0 is perfect, 0.667 a uniform guess, 2.0 the worst attainable. Bounded, so unlike log
    loss no single confident miss can dominate it.
    """
    truth = np.zeros_like(probs)
    truth[np.arange(len(probs)), actuals] = 1.0
    return ((probs - truth) ** 2).sum(axis=1).mean()


def log_loss(probs, actuals):
    """Mean negative log probability of the realised outcome. Proper, lower is better."""
    return -np.log(probs[np.arange(len(probs)), actuals]).mean()


def accuracy(probs, actuals):
    """Share of matches whose modal outcome was the one that happened.

    Not proper, and blind to how the probability mass is spread. Readability only.
    """
    return (probs.argmax(axis=1) == actuals).mean()


def scores(probs, actuals):
    """All three headline metrics for one set of predictions, as a Series."""
    return pd.Series({"brier": brier(probs, actuals),
                      "log loss": log_loss(probs, actuals),
                      "accuracy": accuracy(probs, actuals)}).rename_axis("metric")


def rates(actuals):
    """Observed outcome frequencies, as [home win, draw, away win]."""
    return np.bincount(actuals, minlength=3) / len(actuals)


def calibration(probs, actuals):
    """Mean predicted probability against observed frequency, per outcome.

    Exposes systematic bias that the aggregate scores hide, notably the draw deficit that
    follows from treating the two scorelines as independent.
    """
    return pd.DataFrame({"predicted": probs.mean(axis=0),
                         "actual": rates(actuals)},
                        index=OUTCOMES).rename_axis(index="outcome", columns="share of matches")
