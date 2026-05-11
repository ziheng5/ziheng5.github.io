export const site = {
  title: "Cold Rain's Blog",
  subtitle: "冷雨的博客",
  description: "銀の龍の背に乗って...",
  author: "ColdRain",
  language: "zh-CN",
  avatar: "/images/head.jpg",
  loading: "/images/loading.gif",
  background: ["/images/background3.jpg"],
  since: 2022
};

export const colors = ["#ffa2c4", "#00bcd4", "#03a9f4", "#00a596", "#ff7d73"];

export const menu = [
  { label: "Homepage", icon: "house", theme: "solid", href: "/" },
  { label: "About", icon: "id-card", theme: "solid", href: "/2024/11/26/test/" },
  { label: "Archives", icon: "box-archive", theme: "solid", href: "/archives/" },
  { label: "Categories", icon: "bookmark", theme: "solid", href: "/categories/杂谈/" },
  { label: "Tags", icon: "tags", theme: "solid", href: "/tags/杂谈/" },
  { label: "Links", icon: "link", theme: "solid", href: "/friends/" },
  { label: "Favorites", icon: "star", theme: "solid", href: "/favorites/" }
];

export const iconLinks = [
  { label: "GitHub", icon: "github", theme: "brands", href: "https://github.com/ziheng5" },
  { label: "Bilibili", icon: "bilibili", theme: "brands", href: "https://space.bilibili.com/3546747278198798" },
  { label: "QQ", icon: "qq", theme: "brands", href: "qq://2199325776" },
  { label: "Discord", icon: "discord", theme: "brands", href: "https://discord.gg/htf5PNEX" },
  { label: "Kaggle", icon: "kaggle", theme: "brands", href: "https://www.kaggle.com/ziheng4" }
];

export const friends = [
  { name: "清風之戀", url: "https://blog.qingfengzl.top/", avatar: "https://blog.qingfengzl.top/_astro/matsuri.CKUHXjMu_Z1Oql9k.webp", desc: "清風之戀の空想森林" },
  { name: "SeanDictionary", url: "https://seandictionary.top", avatar: "https://seandictionary.top/wp-content/uploads/2024/09/哭哭_透明.png", desc: "一个苦逼的IoT学生" },
  { name: "594飞飘", url: "https://feipiao594.github.io", avatar: "https://feipiao594.github.io/images/feipiao.gif", desc: "Pray to take back the sunflower that was given out from the bottom of my heart." },
  { name: "Kingpoem", url: "https://kingpoem.github.io", avatar: "https://kingpoem.github.io/assets/avatar.png", desc: "Kingpoem's blog" },
  { name: "William Wei", url: "https://williamwei.top", avatar: "https://williamwei.top/links/avatar.webp", desc: "Everything will be well. Hopefully." },
  { name: "T.本秋", url: "https://blog.texsd.eu.org/", avatar: "https://blog.texsd.eu.org/imgs/avatar_hu_574f01a50f8ac821.webp", desc: "Nothing's true, everything's permitted." },
  { name: "BlueSpace", url: "https://blog.bluespace.ren/", avatar: "https://blog.bluespace.ren/img/soine.png", desc: "欢迎来到一个...常年精神内耗者的空间..._" },
  { name: "Tianyi Lyu", url: "https://lvmolvmo.github.io/", avatar: "https://lvmolvmo.github.io/images/android-chrome-192x192.png", desc: "From blurry images to clear decisions: I connect the dots, literally. ( ͡° ͜ʖ ͡°)" }
];

export const favorites = [
  {
    name: "工具文档",
    items: [
      { name: "Hexo 文档", url: "https://hexo.io/", desc: "快速、简洁且高效的博客框架", icon: "book" },
      { name: "NapCatQQ 文档", url: "https://napneko.github.io/", desc: "现代化的基于 NTQQ 的 Bot 协议端实现", icon: "book" },
      { name: "NoneBot", url: "https://nonebot.dev/", desc: "跨平台 Python 异步机器人框架", icon: "book" },
      { name: "LangChain", url: "https://docs.langchain.com/", desc: "强大的代理工程平台", icon: "book" }
    ]
  },
  {
    name: "实用工具",
    items: [
      { name: "AnyRouter", url: "https://anyrouter.top/", desc: "公益中转站", icon: "car" }
    ]
  }
];
