import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { jsxRenderer } from "hono/jsx-renderer";

import neuralNet from "./routes/neural-net/index";
import simpleChat from "./routes/simple-chat";

const app = new Hono();

app.use("/static/*", serveStatic({ root: "./dist" }));

app.get(
  "*",
  jsxRenderer(({ children }) => {
    return (
      <html lang="en">
        <head>
          <meta charSet="UTF-8" />
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
          />
          <title>Hono JSX Renderer</title>
        </head>
        <body>{children}</body>
      </html>
    );
  }),
);

app.get("/", (c) =>
  c.render(
    <>
      <div id="root">
        {import.meta.env.PROD && (
          <link rel="stylesheet" href="/static/index.css" />
        )}
        {import.meta.env.PROD && (
          <script type="module" src="/static/client.js" />
        )}
        {!import.meta.env.PROD && (
          <script type="module" src="/src/client/index.tsx" />
        )}
      </div>
    </>,
  ),
);

app.notFound((c) =>
  c.render(
    <>
      <div id="root">
        {import.meta.env.PROD && (
          <link rel="stylesheet" href="/static/index.css" />
        )}
        {import.meta.env.PROD && (
          <script type="module" src="/static/client.js" />
        )}
        {!import.meta.env.PROD && (
          <script type="module" src="/src/client/index.tsx" />
        )}
      </div>
    </>,
  ),
);

export const _routes = app.route("/", simpleChat).route("/", neuralNet);

export type App = typeof _routes;
export default app;

if (import.meta.env.PROD) {
  const port = Number(process.env.PORT || 3000);
  serve({ fetch: app.fetch, port }, () => {
    console.log(`Server started on http://localhost:${port}`);
  });
}
