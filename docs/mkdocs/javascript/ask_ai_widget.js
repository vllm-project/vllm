// "Ask AI" assistant for the docs; replaces the RunLLM widget.
document.addEventListener("DOMContentLoaded", function () {
  var script = document.createElement("script");
  script.id = "vllm-ask-ai-widget-script";
  script.src = "https://vllm-docs-qa.vercel.app/widget.js";

  script.setAttribute("data-name", "vLLM");
  script.setAttribute("data-shortcut", "Mod+j"); // cmd-j or ctrl-j to open the widget.
  script.setAttribute("data-position", "bottom-right");
  script.setAttribute("data-offset-x", "20px");
  script.setAttribute("data-offset-y", "120px"); // clear of the version selector.

  script.async = true;
  document.head.appendChild(script);
});
