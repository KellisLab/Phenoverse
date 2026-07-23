document$.subscribe(function () {
  var content = document.querySelector(".md-content");
  if (!content) return;

  var hrs = content.querySelectorAll("hr");
  if (!hrs.length) {
    content.style.removeProperty("--divider-height");
    return;
  }

  var footerHr = hrs[hrs.length - 1];
  var contentTop = content.getBoundingClientRect().top;
  var hrTop = footerHr.getBoundingClientRect().top;
  content.style.setProperty("--divider-height", (hrTop - contentTop) + "px");
});
