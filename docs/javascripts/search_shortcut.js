document$.subscribe(function () {
  var searchForm = document.querySelector('.md-search__form');
  if (searchForm && !searchForm.querySelector('.search-shortcut-hint')) {
    var isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
    var hint = document.createElement('div');
    hint.className = 'search-shortcut-hint';
    hint.textContent = isMac ? '⌘K' : 'Ctrl+K';
    searchForm.appendChild(hint);
  }
});

document.addEventListener('keydown', function (e) {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    var searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
      searchInput.focus();
    }
  }
});
