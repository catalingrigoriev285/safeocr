import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  // keep the dev badge away from the chat sidebar footer
  devIndicators: { position: "bottom-right" },
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
