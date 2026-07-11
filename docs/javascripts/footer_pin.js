document$.subscribe(function () {
  var footer = document.querySelector(".page-footer");
  if (!footer) return;

  var hr = footer.previousElementSibling;
  while (hr && hr.tagName !== "HR") {
    hr = hr.previousElementSibling;
  }
  var target = hr || footer;

  target.style.marginTop = "";
  var rect = footer.getBoundingClientRect();
  var shortfall = window.innerHeight - rect.bottom;
  if (shortfall > 0) {
    var currentMargin = parseFloat(getComputedStyle(target).marginTop) || 0;
    target.style.marginTop = (currentMargin + shortfall) + "px";
  }
});
