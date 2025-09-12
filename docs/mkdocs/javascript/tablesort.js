// https://squidfunk.github.io/mkdocs-material/reference/data-tables/#sortable-tables
document$.subscribe(function() {
  var tables = document.querySelectorAll("article table:not([class])")
  tables.forEach(function(table) {
    new Tablesort(table)
  })
})