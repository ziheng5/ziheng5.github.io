(() => {
  const loading = document.getElementById("loading");
  const main = document.getElementById("main");
  const menu = document.getElementById("menu");
  const mobileTitle = document.querySelector("#mobile-menu .title");
  const mobileItems = document.getElementById("mobile-menu-items");
  const curtain = document.getElementById("menu-curtain");
  const homeWrap = document.getElementById("home-posts-wrap");
  const preview = document.getElementById("preview");
  const previewContent = document.getElementById("preview-content");
  let lastScrollTop = 0;

  function finishLoading() {
    if (loading) loading.hidden = true;
    if (main) {
      main.classList.remove("into-enter-from");
      main.classList.add("into-enter-active");
    }
  }

  function setMobileMenu(open) {
    if (!mobileItems || !curtain || !mobileTitle) return;
    mobileItems.hidden = !open;
    curtain.hidden = !open;
    mobileTitle.setAttribute("aria-expanded", String(open));
  }

  function handleScroll() {
    if (!menu) return;
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    menu.classList.toggle("hidden", scrollTop > lastScrollTop);
    setMobileMenu(false);
    if (homeWrap) {
      menu.classList.toggle("menu-color", scrollTop <= window.innerHeight - 100);
      homeWrap.style.top = scrollTop <= 400 ? `${-scrollTop / 5}px` : "-80px";
    } else {
      menu.classList.remove("menu-color");
    }
    lastScrollTop = Math.max(scrollTop, 0);
  }

  function buildToc() {
    const toc = document.querySelector("#toc .toc");
    const content = document.querySelector(".article-content .content");
    if (!toc || !content) return;
    const headings = Array.from(content.querySelectorAll("h2, h3, h4, h5, h6"));
    if (headings.length === 0) {
      const card = document.getElementById("post-toc-card");
      if (card) card.hidden = true;
      return;
    }
    toc.innerHTML = "";
    headings.forEach((heading) => {
      if (!heading.id) {
        heading.id = heading.textContent.trim().toLowerCase().replace(/\s+/g, "-");
      }
      const li = document.createElement("li");
      li.className = "toc-list-item";
      li.style.paddingLeft = `${Math.max(Number(heading.tagName.slice(1)) - 2, 0)}em`;
      const link = document.createElement("a");
      link.className = "toc-link";
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const offset = 70;
        const top = heading.getBoundingClientRect().top + window.scrollY - offset;
        window.history.pushState(null, "", `#${heading.id}`);
        window.scrollTo({ top, behavior: "smooth" });
      });
      li.appendChild(link);
      toc.appendChild(li);
    });
  }

  function bindSearch() {
    const input = document.getElementById("search-bar");
    const timelines = Array.from(document.querySelectorAll(".timeline"));
    if (!input || timelines.length === 0) return;

    timelines.forEach((item) => {
      item.style.height = "auto";
    });

    input.addEventListener("input", () => {
      const query = input.value.toLowerCase().replace(/\s+/g, "");
      timelines.forEach((item) => {
        const text = `${item.dataset.title || ""}${item.textContent || ""}`.toLowerCase().replace(/\s+/g, "");
        const shouldHide = query.length > 0 && !text.includes(query);
        const isHidden = item.classList.contains("search-hidden");

        item.style.height = `${item.scrollHeight}px`;
        item.offsetHeight;

        if (shouldHide && !isHidden) {
          item.classList.add("search-hidden");
          item.style.height = "0px";
        } else if (!shouldHide && isHidden) {
          item.classList.remove("search-hidden");
          item.style.height = `${item.scrollHeight}px`;
          window.setTimeout(() => {
            if (!item.classList.contains("search-hidden")) item.style.height = "auto";
          }, 360);
        } else if (!shouldHide) {
          item.style.height = "auto";
        } else {
          item.style.height = "0px";
        }
      });
    });
  }

  function bindHomeScroll() {
    const homeInfo = document.getElementById("home-info");
    const homePosts = document.getElementById("home-posts-wrap");
    if (!homeInfo || !homePosts) return;
    homeInfo.addEventListener("click", (event) => {
      event.preventDefault();
      homePosts.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function bindPreview() {
    if (!preview || !previewContent) return;
    document.querySelectorAll(".content img").forEach((img) => {
      img.addEventListener("click", () => {
        previewContent.src = img.currentSrc || img.src;
        preview.hidden = false;
      });
    });
    preview.addEventListener("click", () => {
      preview.hidden = true;
      previewContent.removeAttribute("src");
    });
  }

  window.addEventListener("load", finishLoading);
  window.addEventListener("scroll", handleScroll, { passive: true });
  mobileTitle?.addEventListener("click", () => setMobileMenu(mobileItems?.hidden));
  curtain?.addEventListener("click", () => setMobileMenu(false));
  document.addEventListener("DOMContentLoaded", () => {
    finishLoading();
    handleScroll();
    bindHomeScroll();
    buildToc();
    bindSearch();
    bindPreview();
  });
})();
