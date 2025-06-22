keyboard$.subscribe(key => {
  if (key.mode === 'global' && key.type === 'k' && (key.ctrlKey || key.metaKey)) {
    key.claim();
    query$.focus();
    query$.select();
  }
})
