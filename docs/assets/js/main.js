"use strict";

const copyButton = document.getElementById("copy-citation");
const citation = document.getElementById("citation");
const copyStatus = document.getElementById("copy-status");
if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      if (!navigator.clipboard) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(citation.textContent);
      copyStatus.textContent = "Citation copied.";
    } catch {
      // Leave the citation selected for manual copying when clipboard access is blocked.
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(citation);
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Citation selected. Press Ctrl+C or ⌘C to copy.";
    }
  });
}


// Keep the slide's poster until the local MP4 is available. Copying the file
// into assets/videos/crossword-demo.mp4 and refreshing is enough to enable it.
const demo = document.getElementById("crossword-demo");
if (demo) {
  const video = document.getElementById("crossword-video");
  const preview = document.getElementById("crossword-preview");
  async function loadDemo() {
    try {
      const response = await fetch(demo.dataset.videoSrc, { method: "HEAD", cache: "no-store" });
      if (!response.ok) return;
      video.addEventListener("error", () => {
        video.hidden = true;
        preview.hidden = false;
      });
      video.src = demo.dataset.videoSrc;
      video.hidden = false;
      preview.hidden = true;
    } catch {
      // The poster remains readable offline and before the video is supplied.
    }
  }
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        loadDemo();
      }
    }, { rootMargin: "300px" });
    observer.observe(demo);
  } else {
    loadDemo();
  }
}
