import { colors } from "./site";

export type BlogPost = {
  title: string;
  date: Date;
  description?: string;
  tags: string[];
  categories: string[];
  slug: string;
  path: string;
  pinned: number;
  Content: any;
};

const modules = import.meta.glob("../posts/*.md", { eager: true }) as Record<string, any>;
const rawModules = import.meta.glob("../posts/*.md", {
  eager: true,
  query: "?raw",
  import: "default"
}) as Record<string, string>;

function list(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(String);
  return [String(value)];
}

function rawDate(file: string): string | undefined {
  const source = rawModules[file] || "";
  return source.match(/^date:\s*["']?(.+?)["']?\s*$/m)?.[1];
}

function parseDate(value: unknown, raw?: string): Date {
  const source = raw || (value instanceof Date ? value.toISOString() : String(value));
  const match = source.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?/);
  if (match) {
    const [, year, month, day, hour = "0", minute = "0", second = "0"] = match;
    return new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second)
    );
  }
  if (value instanceof Date) return value;
  return new Date(String(value).replace(" ", "T"));
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function formatDate(date: Date): string {
  return `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()}`;
}

export function postPath(date: Date, slug: string): string {
  return `/${date.getFullYear()}/${pad(date.getMonth() + 1)}/${pad(date.getDate())}/${slug}/`;
}

export function tagColor(index: number): string {
  return colors[index % colors.length];
}

export function slugify(value: string): string {
  return encodeURIComponent(value);
}

export function plainSlugFromModulePath(path: string): string {
  const file = path.split("/").pop() || "";
  return decodeURIComponent(file.replace(/\.md$/, ""));
}

export function getPosts(): BlogPost[] {
  return Object.entries(modules)
    .map(([file, mod]) => {
      const frontmatter = mod.frontmatter || {};
      const slug = plainSlugFromModulePath(file);
      const date = parseDate(frontmatter.date, rawDate(file));
      return {
        title: frontmatter.title || slug,
        date,
        description: frontmatter.description,
        tags: list(frontmatter.tags),
        categories: list(frontmatter.categories),
        slug,
        path: postPath(date, slug),
        pinned: Number(frontmatter.pinned || 0),
        Content: mod.Content
      };
    })
    .filter((post) => !Number.isNaN(post.date.getTime()))
    .sort((a, b) => b.pinned - a.pinned || b.date.getTime() - a.date.getTime());
}

export function uniqueValues(posts: BlogPost[], key: "tags" | "categories"): string[] {
  return Array.from(new Set(posts.flatMap((post) => post[key]))).sort((a, b) => a.localeCompare(b, "zh-CN"));
}

export function postsByValue(posts: BlogPost[], key: "tags" | "categories", value: string): BlogPost[] {
  return posts.filter((post) => post[key].includes(value));
}

export function excerpt(post: BlogPost): string {
  return post.description || "";
}
