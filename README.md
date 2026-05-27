# Academic Project Page

This is a lightweight ML/AI paper project page template inspired by common academic project pages.

## Local Preview

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000` in your browser.

## What To Replace

- Edit `index.html` for the paper title, authors, abstract, links, sections, and BibTeX.
- Put figures in `static/images/`.
- Put videos in `static/videos/`.
- Put PDFs in `static/pdfs/`.

## GitHub Pages

After creating a new GitHub repository:

```bash
git init
git add .
git commit -m "Initial project page"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

Then enable GitHub Pages from `Settings -> Pages -> Deploy from a branch -> main / root`.
