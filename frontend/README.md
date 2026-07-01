# Unified Multi-Agent Data Analysis Interface

This is a project designed as a unified interface for users to interact with a multi-agent data analysis workflow. It focuses on visualizing complex analysis processes through plotted graphs and organized agent insights.

## Features

- **Multi-turn Conversation:** Intuitive chat interface for continuous dialogue with agents to refine analysis goals.
- **Real-time Graph Structure Update:** Dynamic visualization of the multi-agent workflow structure as it evolves during the session.
- **Grid-Layout Insights:** Agent responses and generated analysis insights are presented in a structured grid form for easy comparison and review.

## Getting Started

First, install required dependencies:

```bash
npm i
```

Create a `.env` file in the root directory:

```dotenv
UPSTREAM_URL=http://0.0.0.0:8000
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
