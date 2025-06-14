/**
 * feedback.js
 *
 * Enhances MkDocs Material docs pages by:
 *
 * 1. Adding a "💡 Question? Give us feedback" link
 *    below the "✏️ Suggest edit on GitHub" button.
 *
 *    - The link opens a GitHub issue with a template,
 *      auto-filled with the current page URL and path.
 *
 * 2. Ensuring the edit button opens in a new tab
 *    with target="_blank" and rel="noopener".
 */
document.addEventListener("DOMContentLoaded", function () {
  const url = window.location.href;
  const page = document.body.dataset.mdUrl || location.pathname;

  const feedbackLink = document.createElement("a");
  feedbackLink.href = `https://github.com/vllm-project/vllm/issues/new?template=100-documentation.yml&title=${encodeURIComponent(
    `[Docs] Feedback for \`${page}\``
  )}&body=${encodeURIComponent(`📄 **Reference:**\n${url}\n\n📝 **Feedback:**\n_Your response_`)}`;
  feedbackLink.target = "_blank";
  feedbackLink.rel = "noopener";
  feedbackLink.textContent = "💡 Question? Give us feedback";
  feedbackLink.className = "doc-feedback-link";

  const editButton = document.querySelector('.md-content__button[href*="edit"]');
  if (editButton && editButton.parentNode) {
    const clonedEditButton = editButton.cloneNode(true);
    clonedEditButton.setAttribute('target', '_blank');
    clonedEditButton.setAttribute('rel', 'noopener');

    const container = document.createElement("div");
    container.className = "doc-feedback-wrapper";
    container.appendChild(clonedEditButton);
    container.appendChild(feedbackLink);

    editButton.replaceWith(container);
  }
});
