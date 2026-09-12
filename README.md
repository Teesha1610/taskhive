# 🐙 GitHub Analytics Dashboard

A beautiful Python dashboard that visualizes any GitHub user's public data —
repos, languages, stars, forks, commit activity, and more.

## Stack
- **Streamlit** — Python web app framework (runs in your browser)
- **Plotly** — interactive charts
- **GitHub REST API** — completely free, no account needed for public data

---

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```
Your browser opens automatically at `http://localhost:8501`

### 3. Enter any GitHub username
Type a username in the sidebar and click **Analyze →**

---

## Charts included
| Chart | What it shows |
|---|---|
| Top repos by stars | Bar chart of your most-starred repos |
| Language breakdown | Donut chart of bytes written per language |
| Commit activity | Daily commit counts over the last 90 days |
| Stars vs Forks | Bubble scatter — repo popularity map |
| Activity breakdown | Push, PR, issue, fork, comment counts |
| Repo timeline | When each repo was created, sized by stars |
| Full repo table | Sortable table of all public repos |

---

## Rate limits (GitHub API)

| Mode | Limit |
|---|---|
| No token | 60 requests/hour |
| Free personal token | 5,000 requests/hour |

To get a free token: https://github.com/settings/tokens
(Select "No expiration", no scopes needed for public data)

Paste it into the **Personal access token** field in the sidebar.

---

## VS Code tip
Install the **Streamlit** extension for VS Code to get a "Run Streamlit" button
directly in the editor. Or just use the integrated terminal.
