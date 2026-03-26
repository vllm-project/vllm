// Add RunLLM widget
document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.type = "module";
    script.id = "runllm-widget-script"
  
    script.src = "https://widget.runllm.com";
  
    script.setAttribute("version", "stable");
    script.setAttribute("runllm-keyboard-shortcut", "Mod+j"); // cmd-j or ctrl-j to open the widget.
    script.setAttribute("runllm-name", "vLLM");
    script.setAttribute("runllm-position", "BOTTOM_RIGHT");
    script.setAttribute("runllm-position-y", "120px");
    script.setAttribute("runllm-position-x", "20px");
    script.setAttribute("runllm-assistant-id", "207");
  
    script.async = true;
    document.head.appendChild(script);
  });
