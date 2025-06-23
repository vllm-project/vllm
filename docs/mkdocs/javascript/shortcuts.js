document.addEventListener('keydown', function (event) {
  if (event.key === '/') {
    // we actually don't want to bind this key on docs given that it uses builtin mkdocs search.
    event.preventDefault();
  } elif ((event.altKey || event.metaKey) && event.key === 'k') {
    event.preventDefault();
    const query = document.querySelector('[data-md-component="search-query"]');
    if (query) {
      query.focus();
      query.select();
    }
  }
});
