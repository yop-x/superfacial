document.getElementById("save-btn").addEventListener("click", () => {
  const results = [];

  document.querySelectorAll('input[type="radio"]:checked').forEach((input) => {
    results.push({
      name: input.name,     // "face-12" (or whatever your group name is)
      label: input.value    // "yes" or "no"
    });
  });

  if (results.length === 0) {
    alert("No selections yet. Pick Yes/No for at least one face.");
    return;
  }

  fetch("http://localhost:8000/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results)
  })
  .then(async (res) => {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    console.log("BACKEND RESPONSE:", data);
    alert(`Saved successfully! (${data.count} labels)`);
  })
  .catch((err) => {
    console.error(err);
    alert("Save failed: " + err.message);
  });
});
