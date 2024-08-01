function fetchGeminiOutput() {
  fetch("/gemini")
    .then((response) => response.text())
    .then((data) => {
      document.getElementById("gemini-output").innerText = data;
    });
}
