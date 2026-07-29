# gaba5.github.io

Sebastien's personal site, live at https://sebastienlorentz.com. Hand-written
static HTML/CSS/JS on GitHub Pages. No build step, no generator, `.nojekyll`.

## Structure
- `index.html` - hero, plus a few recent items from each section.
- `projects/`, `experiments/` - one folder per item, each with its own page.
  `projects/template.html` is the starting point for a new page.
- Every item appears as a `.project-item` card (title, desc, `.item-tag` chips,
  date) in its section index, and again in `index.html` if it should show under
  "Recent". Those two section indexes are the source of truth: `tags.html` fetches
  and parses them at runtime to build tag collections, and `assets/js/tags.js`
  makes every chip clickable by delegation.
- `assets/css/main.css` - the whole look, driven by CSS custom properties.

Adding an item = new folder + page from the template, then paste its card into the
section index (and `index.html` if recent). Keep the two copies in sync.

## The idea vault
Sebastien's ideas live in a separate Claude Code project, an Obsidian vault at
`C:\Users\sebas\iCloudDrive\iCloud~md~obsidian\Ideas_`. Notes are in `Ideas/`,
finished ones in `Archive/done/`. Read it freely; that project owns those files, so
do not write to it.

Relevant frontmatter on an idea note:
- `for_site: true` - this idea is earmarked to be built here. These are the
  backlog: when he asks what is queued for the site, list them.
- `kind: project | experiment` - which section it belongs in.
- `site: /projects/CPM/cpm.html` - already live here.
- `status`, `effort`, `tags`, plus a researched body with first steps and resources.

When starting one of these, the note's body is the spec. He writes the prose and
uploads the figures and code himself; help with the page only as far as he asks.
The vault side handles its own bookkeeping (status, archiving, recording the URL).
