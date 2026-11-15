/**
 * edit_and_feedback.js
 *
 * Enhances MkDocs Material docs pages by:
 *
 * 1. Adding a "Question? Give us feedback" link
 *    below the "Edit" button.
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
  feedbackLink.title = "Provide feedback";
  feedbackLink.className = "md-content__button";
  feedbackLink.innerHTML = `
  <svg
    xmlns="http://www.w3.org/2000/svg"
    height="24px"
    viewBox="0 -960 960 960"
    width="24px"
    fill="currentColor"
  >
    <path d="M280-280h280v-80H280v80Zm0-160h400v-80H280v80Zm0-160h400v-80H280v80Zm-80 480q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z"/>
  </svg>
`;

  const editButton = document.querySelector('.md-content__button[href*="edit"]');

  if (editButton && editButton.parentNode) {
    editButton.insertAdjacentElement("beforebegin", feedbackLink);

    editButton.setAttribute("target", "_blank");
    editButton.setAttribute("rel", "noopener");
  }
});
