// UI 操作函数
function appendMessage(sender, text) {
    const chatbox = document.getElementById("chatbot");
    const div = document.createElement("div");
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
  }
  