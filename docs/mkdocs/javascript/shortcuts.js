// Enable vertical scrolling with arrow keys
keyboard$.subscribe(function (key) {
  const scrollStep = window.innerHeight * 0.1; // 10% of viewport height
  if (key.mode === "global") {
    switch (key.type) {
      case "ArrowUp":
        window.scrollBy({top: -scrollStep, behavior: "smooth"});
        key.claim();
        break;
      case "ArrowDown":
        window.scrollBy({top: scrollStep, behavior: "smooth"});
        key.claim();
        break;
    }
  }
})
