/**
 * slack_and_forum.js
 *
 * Adds a custom Slack and Forum button to the MkDocs Material header.
 *
 */

window.addEventListener('DOMContentLoaded', () => {
  const headerInner = document.querySelector('.md-header__inner');

  if (headerInner) {
    const slackButton = document.createElement('button');
    slackButton.className = 'slack-button';
    slackButton.title = 'Join us on Slack';
    slackButton.style.border = 'none';
    slackButton.style.background = 'transparent';
    slackButton.style.cursor = 'pointer';

    slackButton.innerHTML = `
      <img src="https://a.slack-edge.com/80588/marketing/img/icons/icon_slack_hash_colored.png" 
           style="height: 1.1rem;" 
           alt="Slack">
    `;

    slackButton.addEventListener('click', () => {
      window.open('https://slack.vllm.ai', '_blank', 'noopener');
    });

    const forumButton = document.createElement('button');
    forumButton.className = 'forum-button';
    forumButton.title = 'Join the Forum';
    forumButton.style.border = 'none';
    forumButton.style.background = 'transparent';
    forumButton.style.cursor = 'pointer';

    forumButton.innerHTML = `
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 -960 960 960"
        fill="currentColor"
      >
        <path d="M817.85-198.15 698.46-317.54H320q-24.48 0-41.47-16.99T261.54-376v-11.69h424.61q25.39 0 43.47-18.08 18.07-18.08 18.07-43.46v-268.92h11.69q24.48 0 41.47 16.99 17 16.99 17 41.47v461.54ZM179.08-434.69l66.84-66.85h363.31q10.77 0 17.69-6.92 6.93-6.92 6.93-17.69v-246.77q0-10.77-6.93-17.7-6.92-6.92-17.69-6.92H203.69q-10.77 0-17.69 6.92-6.92 6.93-6.92 17.7v338.23Zm-36.93 89.46v-427.69q0-25.39 18.08-43.46 18.08-18.08 43.46-18.08h405.54q25.39 0 43.46 18.08 18.08 18.07 18.08 43.46v246.77q0 25.38-18.08 43.46-18.07 18.07-43.46 18.07H261.54L142.15-345.23Zm36.93-180.92V-797.54v271.39Z"/>
      </svg>
    `;

    forumButton.addEventListener('click', () => {
      window.open('https://discuss.vllm.ai/', '_blank', 'noopener');
    });

    const githubSource = document.querySelector('.md-header__source');
    if (githubSource) {
      githubSource.parentNode.insertBefore(slackButton, githubSource.nextSibling);
      githubSource.parentNode.insertBefore(forumButton, slackButton.nextSibling);
    }
  }
});
