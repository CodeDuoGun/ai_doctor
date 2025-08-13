// UI 操作函数// UI 操作函数
function appendMessage(sender, text, type = "text") {
  const chatbox = document.getElementById("chatbot");

  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender === "user" ? "user" : "bot");

  const avatar = document.createElement("img");
  avatar.classList.add("avatar");
  avatar.src = sender === "user" ? "/static/user.png" : "/static/ai_girl.png";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  if (type === "text") {
    bubble.textContent = text;
  } else if (type === "image") {
    const img = document.createElement("img");
    img.src = text;
    img.style.maxWidth = "200px";
    img.style.borderRadius = "8px";
    bubble.appendChild(img);
  }

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(bubble);
  chatbox.appendChild(messageDiv);

  chatbox.scrollTop = chatbox.scrollHeight;
}

// 发送文字
document.getElementById("btn-send").addEventListener("click", () => {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (text) {
    appendMessage("user", text);
    input.value = "";
    // TODO: 这里可以发送到后端
  }
});



  