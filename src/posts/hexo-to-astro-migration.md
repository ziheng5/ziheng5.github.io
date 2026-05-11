---
title: 从 Hexo 迁移到 Astro：复刻主题与重新部署博客
date: 2026-05-11 19:30:00
tags:
    - Astro
    - Hexo
    - GitHub Pages
    - Cloudflare
categories:
    - 食用指南
description: |
    记录一次将 Hexo 博客迁移到 Astro 的过程：从内容迁移、主题复刻、路由兼容，到 GitHub Pages 与 Cloudflare 自定义域名部署。
---

## 0. 前言

这个博客原本基于 Hexo 构建，主题使用的是 `particlex`。Hexo 的优点是成熟、插件多、写作体验直接；但在长期维护时，我也逐渐希望把博客改成更容易组件化、更容易自己控制页面结构的形式。

于是这次尝试用 Astro 复刻原本的 Hexo 主题，并尽可能保留：

- 原来的文章 Markdown 内容；
- 原来的图片资源路径；
- 原来的首页、文章页、归档页、分类页、标签页；
- 原来的 `/year/month/day/slug/` 文章链接；
- 原来的 GitHub Pages + Cloudflare 自定义域名部署方式。

本文记录完整迁移过程，方便以后维护或复现。

## 1. 为什么选择 Astro

Astro 很适合静态博客，原因主要有几个：

- 默认输出静态 HTML，适合 GitHub Pages；
- 组件化比 Hexo EJS 模板更直观；
- Markdown 支持很好；
- 可以逐步加交互，不需要把整个博客变成 SPA；
- 对自定义路由、布局、样式控制更细。

这次迁移不是直接套一个现成 Astro 主题，而是把原来的 Hexo 主题结构拆开，再用 Astro 重新实现。

## 2. 原 Hexo 项目结构

原来的 Hexo 博客大致结构如下：

```text
HexoBlog/ziheng5.github.io/
├── _config.yml
├── _config.yun.yml
├── source/
│   ├── _posts/
│   ├── images/
│   ├── friends/
│   ├── favorites/
│   └── _data/
└── themes/
    └── particlex/
        ├── layout/
        └── source/
```

其中最重要的是：

- `source/_posts/`：文章；
- `source/images/`：图片；
- `themes/particlex/layout/`：EJS 模板；
- `themes/particlex/source/css/main.css`：主题主样式；
- `themes/particlex/source/js/`：主题脚本。

迁移时不要急着删旧项目，建议先在同级目录新建一个 Astro 项目，确认复刻成功后再部署。

## 3. 新建 Astro 项目

可以新建一个目录，例如：

```bash
mkdir AstroBlog
cd AstroBlog
```

初始化依赖时，可以使用 `pnpm`：

```bash
pnpm init
pnpm add astro @astrojs/sitemap remark-math rehype-katex
```

`package.json` 中保留三个常用脚本：

```json
{
  "scripts": {
    "dev": "ASTRO_TELEMETRY_DISABLED=1 astro dev --host 0.0.0.0",
    "build": "ASTRO_TELEMETRY_DISABLED=1 astro build",
    "preview": "ASTRO_TELEMETRY_DISABLED=1 astro preview --host 0.0.0.0"
  }
}
```

其中 `ASTRO_TELEMETRY_DISABLED=1` 不是必须的，但在一些受限环境里可以避免 Astro telemetry 写入用户目录导致构建失败。

## 4. Astro 目录设计

这次复刻后的目录大致如下：

```text
AstroBlog/
├── astro.config.mjs
├── package.json
├── pnpm-lock.yaml
├── public/
│   ├── CNAME
│   ├── .nojekyll
│   ├── css/
│   ├── js/
│   └── images/
└── src/
    ├── components/
    ├── layouts/
    ├── lib/
    ├── pages/
    └── posts/
```

这里没有使用 Astro Content Collections，而是用 `import.meta.glob()` 直接读取 `src/posts/*.md`。这样做的好处是迁移成本低，文章格式可以尽量贴近原 Hexo。

## 5. 迁移文章

将 Hexo 的文章复制到 Astro：

```bash
cp HexoBlog/ziheng5.github.io/source/_posts/*.md AstroBlog/src/posts/
```

文章 frontmatter 可以继续保持 Hexo 风格：

```md
---
title: 文章标题
date: 2026-05-11 19:30:00
tags:
    - Astro
    - Hexo
categories:
    - 食用指南
description: |
    首页文章卡片显示的摘要。
---

正文内容...
```

需要注意：

- `date` 建议使用 `YYYY-MM-DD HH:mm:ss`；
- 文件名会作为文章 URL 的 `slug`；
- `tags` 和 `categories` 建议使用数组格式；
- `description` 会显示在首页文章卡片中。

例如：

```text
src/posts/hexo-to-astro-migration.md
```

会生成类似：

```text
/2026/05/11/hexo-to-astro-migration/
```

## 6. 迁移图片资源

Hexo 中的图片通常在：

```text
source/images/
```

Astro 中可以放到：

```text
public/images/
```

复制命令：

```bash
cp -R HexoBlog/ziheng5.github.io/source/images/. AstroBlog/public/images/
```

文章中推荐使用绝对路径：

```md
![示例图片](/images/example/pic.png)
```

如果旧文章中写的是：

```md
![示例图片](./images/example/pic.png)
```

在 Astro 中可能会被解析成相对于当前 Markdown 文件的路径，从而导致构建失败。可以批量替换：

```bash
perl -pi -e 's#\]\(\./images/#](/images/#g' src/posts/*.md
```

## 7. 迁移主题样式

原 Hexo 主题的样式文件可以先直接复用：

```bash
cp HexoBlog/ziheng5.github.io/themes/particlex/source/css/main.css AstroBlog/public/css/main.css
```

然后额外创建一个覆盖样式：

```text
public/css/astro-overrides.css
```

建议以后尽量改 `astro-overrides.css`，少直接改 `main.css`。原因是 `main.css` 来自旧主题，里面有很多全局选择器，直接改容易影响多个页面。

在基础布局中引入：

```astro
<link rel="stylesheet" href="/css/main.css" />
<link rel="stylesheet" href="/css/astro-overrides.css" />
```

## 8. 文章数据读取

可以在 `src/lib/posts.ts` 中统一读取文章：

```ts
const modules = import.meta.glob("../posts/*.md", { eager: true });
```

读取后整理成统一结构：

```ts
export type BlogPost = {
  title: string;
  date: Date;
  description?: string;
  tags: string[];
  categories: string[];
  slug: string;
  path: string;
  Content: any;
};
```

这里有一个细节：Astro 解析 frontmatter 的日期时可能受到时区影响，导致原本 `2024-11-26 22:43:45` 的文章在 URL 中变成 `2024/11/27`。

为了保持 Hexo URL 兼容，最好从 Markdown 原始文本中读取 `date:` 字符串，再手动解析成年月日。

这样旧链接可以继续保持：

```text
/2024/11/26/test/
```

而不是因为时区偏移变成：

```text
/2024/11/27/test/
```

## 9. 实现 Hexo 风格文章路由

Astro 中可以使用动态路由：

```text
src/pages/[year]/[month]/[day]/[slug]/index.astro
```

在 `getStaticPaths()` 中为每篇文章生成路径：

```ts
export function getStaticPaths() {
  return getPosts().map((post) => {
    const [year, month, day] = post.path.split("/").filter(Boolean);
    return {
      params: { year, month, day, slug: post.slug },
      props: { post }
    };
  });
}
```

这样就能得到和 Hexo 类似的静态路径。

## 10. 实现首页、归档、分类和标签

首页可以读取所有文章，按日期倒序排列：

```ts
const posts = getPosts();
const pageSize = 10;
const pagePosts = posts.slice(0, pageSize);
```

分页路由可以放在：

```text
src/pages/page/[page]/index.astro
```

归档页：

```text
src/pages/archives/index.astro
```

分类页：

```text
src/pages/categories/index.astro
src/pages/categories/[category].astro
```

标签页：

```text
src/pages/tags/index.astro
src/pages/tags/[tag].astro
```

分类和标签都可以通过文章 frontmatter 计算：

```ts
export function uniqueValues(posts, key) {
  return Array.from(new Set(posts.flatMap((post) => post[key])));
}
```

## 11. 代码高亮和数学公式

Astro 默认使用 Shiki 做代码高亮。为了支持数学公式，可以加入：

```bash
pnpm add remark-math rehype-katex
```

`astro.config.mjs`：

```js
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

export default defineConfig({
  site: "https://coldrain.top",
  integrations: [sitemap()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: "github-light",
      wrap: true
    }
  }
});
```

还需要引入 KaTeX 样式：

```astro
<link rel="stylesheet" href="https://s4.zstatic.net/ajax/libs/KaTeX/0.16.9/katex.min.css" />
```

Astro 生成的代码块会带有 `data-language`，可以用 CSS 显示语言标签：

```css
.content .astro-code[data-language]::before {
  content: attr(data-language);
  position: absolute;
  top: 0;
  left: 24px;
}
```

## 12. 前端交互

原 Hexo 主题里有一些 Vue 交互，比如：

- 加载动画；
- 移动端菜单；
- 首页圆形 info 点击下滑；
- 归档搜索；
- 图片预览；
- 文章目录。

迁移到 Astro 后，不一定需要继续引入 Vue。简单交互可以写在：

```text
public/js/site.js
```

例如首页圆形 info 平滑滚动：

```js
function bindHomeScroll() {
  const homeInfo = document.getElementById("home-info");
  const homePosts = document.getElementById("home-posts-wrap");
  if (!homeInfo || !homePosts) return;

  homeInfo.addEventListener("click", (event) => {
    event.preventDefault();
    homePosts.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}
```

文章目录点击也可以改成平滑滚动：

```js
link.addEventListener("click", (event) => {
  event.preventDefault();
  const offset = 70;
  const top = heading.getBoundingClientRect().top + window.scrollY - offset;
  window.history.pushState(null, "", `#${heading.id}`);
  window.scrollTo({ top, behavior: "smooth" });
});
```

## 13. 本地运行和构建

开发预览：

```bash
pnpm dev
```

默认地址：

```text
http://localhost:4321/
```

构建：

```bash
pnpm build
```

构建结果在：

```text
dist/
```

如果 `pnpm build` 通过，说明静态站点可以部署。

## 14. 部署到 GitHub Pages

原本 Hexo 常见部署方式是：

```bash
hexo clean
hexo generate
hexo deploy
```

也就是把生成后的静态文件推送到 GitHub Pages 分支。

Astro 更推荐使用 GitHub Actions 自动构建部署。项目根目录创建：

```text
.github/workflows/deploy.yml
```

内容如下：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main, master, hexo]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v6

      - name: Install, build, and upload
        uses: withastro/action@v6
        with:
          package-manager: pnpm@latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v5
```

然后到 GitHub 仓库设置：

```text
Settings -> Pages -> Build and deployment -> Source
```

选择：

```text
GitHub Actions
```

之后每次 push 到对应分支，GitHub Actions 会自动：

1. 安装依赖；
2. 执行 `pnpm build`；
3. 上传 `dist/`；
4. 部署到 GitHub Pages。

## 15. 关于 `site` 和 `base`

如果仓库是：

```text
https://github.com/ziheng5/ziheng5.github.io
```

这是 GitHub Pages 的用户站点仓库，网站部署在根路径：

```text
https://ziheng5.github.io/
```

这种情况不需要设置 `base`。

如果使用自定义域名：

```text
https://coldrain.top/
```

则 `astro.config.mjs` 应该写：

```js
export default defineConfig({
  site: "https://coldrain.top"
});
```

不要写：

```js
base: "/ziheng5.github.io"
```

只有当项目部署到类似下面这种子路径时，才需要 `base`：

```text
https://username.github.io/my-blog/
```

## 16. 配置自定义域名

如果想继续使用：

```text
https://coldrain.top/
```

需要在 Astro 项目的 `public/` 目录中添加：

```text
public/CNAME
```

文件内容只有一行：

```text
coldrain.top
```

构建后它会被复制到：

```text
dist/CNAME
```

同时建议添加：

```text
public/.nojekyll
```

这样 GitHub Pages 不会把站点当作 Jekyll 项目处理。

## 17. Cloudflare 配置注意事项

如果原来已经能通过 Cloudflare 访问 Hexo 站点，那么迁移到 Astro 后，DNS 大概率不用大改。因为站点源头仍然是 GitHub Pages，只是生成器从 Hexo 换成 Astro。

需要确认几个点：

### 17.1 GitHub Pages 设置

在 GitHub 仓库：

```text
Settings -> Pages
```

确认：

```text
Custom domain: coldrain.top
Enforce HTTPS: enabled
```

如果刚设置完域名，`Enforce HTTPS` 可能要等一段时间才能开启。

### 17.2 Cloudflare DNS

根域名 `coldrain.top` 可以使用 GitHub Pages 的 A 记录：

```text
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

如果需要 IPv6，也可以配置 AAAA 记录：

```text
2606:50c0:8000::153
2606:50c0:8001::153
2606:50c0:8002::153
2606:50c0:8003::153
```

如果使用 `www.coldrain.top`，可以配置：

```text
www CNAME ziheng5.github.io
```

Cloudflare 对根域名也支持 CNAME Flattening，所以如果你原来是让 `coldrain.top` 指向 `ziheng5.github.io`，通常也可以继续使用。

### 17.3 SSL/TLS

Cloudflare 的 SSL/TLS 模式建议使用：

```text
Full
```

或：

```text
Full (strict)
```

不要使用 `Flexible`，否则容易出现重定向循环或 HTTPS 状态异常。

## 18. 推送迁移后的项目

确认本地构建成功：

```bash
pnpm build
```

然后把 Astro 项目作为 GitHub Pages 仓库根目录提交。

也就是说，仓库根目录应该直接看到：

```text
astro.config.mjs
package.json
pnpm-lock.yaml
src/
public/
.github/
```

提交：

```bash
git add .
git commit -m "Migrate blog from Hexo to Astro"
git push origin hexo
```

如果你的 GitHub Pages 部署分支改成了 `main` 或 `master`，就推对应分支。

## 19. 常见问题

### 19.1 图片找不到

优先检查 Markdown 图片路径。

推荐：

```md
![pic](/images/demo/pic.png)
```

不推荐：

```md
![pic](./images/demo/pic.png)
```

### 19.2 URL 日期不对

检查 `date:` 是否被 Astro 按 UTC 解析。为兼容 Hexo，最好从 Markdown 原始 frontmatter 字符串解析日期。

### 19.3 代码块语言没有显示

Astro/Shiki 会在代码块上生成 `data-language`，用 CSS 的 `::before` 即可显示。

### 19.4 公式渲染 warning

如果数学公式中有中文，KaTeX 可能给出 warning。只要构建成功，通常不影响页面输出。

### 19.5 GitHub Pages 404

检查：

- GitHub Pages Source 是否选择 GitHub Actions；
- Actions 是否执行成功；
- `public/CNAME` 是否存在；
- `astro.config.mjs` 的 `site` 是否正确；
- 自定义域名是否在 GitHub Pages 设置中保存；
- Cloudflare DNS 是否指向 GitHub Pages。

## 20. 总结

这次迁移的核心不是简单换一个框架，而是把博客拆成了几个更容易维护的部分：

- 文章：`src/posts/`
- 站点配置：`src/lib/site.ts`
- 页面：`src/pages/`
- 组件：`src/components/`
- 样式覆盖：`public/css/astro-overrides.css`
- 交互：`public/js/site.js`
- 部署：`.github/workflows/deploy.yml`

这样以后写文章、改样式、加页面、调整部署都比较清晰。

从最终效果看，Astro 可以比较完整地复刻原 Hexo 主题，同时也让后续自定义变得更灵活。

## 参考资料

- [Astro GitHub Pages 部署文档](https://docs.astro.build/en/guides/deploy/github/)
- [GitHub Pages 自定义域名文档](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site)
