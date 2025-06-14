/**
 *  edit.js
 *  Ensuring the edit button opens in a new tab
 *  with target="_blank" and rel="noopener".
 */
document.addEventListener("DOMContentLoaded", function () {
  const editButton = document.querySelector('.md-content__button[href*="edit"]');
  if (editButton) {
    editButton.setAttribute('target', '_blank');
    editButton.setAttribute('rel', 'noopener');
  }
});
