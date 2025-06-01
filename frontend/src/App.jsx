import React, { useState, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    fetch(`${API_BASE}/collections`)
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setCollections(data.collections);
          if (data.collections.length > 0) {
            setSelectedCollection(data.collections[0].name);
          }
        }
      });
  }, []);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first.");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setUploadMessage(data.message);
        const collectionsRes = await fetch(`${API_BASE}/collections`);
        const collectionsData = await collectionsRes.json();
        if (collectionsData.success) {
          setCollections(collectionsData.collections);
          setSelectedCollection(data.collection_name);
        }
      } else {
        setUploadMessage("Upload failed: " + data.detail);
      }
    } catch (err) {
      setUploadMessage("Upload error: " + err.message);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return alert("Please enter a question.");
    if (!selectedCollection) return alert("Please select a collection.");

    setLoading(true);
    setAnswer("");
    setContext("");

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: selectedCollection }),
      });
      const data = await res.json();
      if (res.ok) {
        setAnswer(data.answer);
        setContext(data.context);
      } else {
        setAnswer("Error: " + (data.detail || "Unknown error"));
        setContext("");
      }
    } catch (err) {
      setAnswer("Error: " + err.message);
      setContext("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-white to-purple-100 flex items-center justify-center px-4 py-10">
      <div className="max-w-5xl w-full bg-white rounded-3xl shadow-2xl p-10 space-y-12 font-sans">
        <h1 className="text-5xl font-black text-center text-indigo-700 tracking-wide drop-shadow">
          AskMyNotes AI 📘
        </h1>

        <section className="space-y-6">
          <h2 className="text-2xl font-semibold text-gray-700 border-b border-indigo-200 pb-2">
            📄 Upload Your Document
          </h2>
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            <input
              type="file"
              accept=".txt,.pdf"
              onChange={handleFileChange}
              className="file:mr-5 file:py-3 file:px-6 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200 transition file:cursor-pointer w-full"
            />
            <button
              onClick={handleUpload}
              className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg shadow-md hover:bg-indigo-700 transition font-semibold w-full sm:w-auto justify-center"
            >
              ⬆️ Upload
            </button>
          </div>
          {uploadMessage && (
            <p className="text-indigo-600 font-medium text-sm mt-1 select-text">
              {uploadMessage}
            </p>
          )}
        </section>

        <section className="space-y-6">
          <h2 className="text-2xl font-semibold text-gray-700 border-b border-indigo-200 pb-2">
            ❓ Ask a Question
          </h2>

          <div>
            <label className="block text-gray-600 font-medium mb-2">
              Select Document Collection:
            </label>
            <select
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              className="w-full rounded-lg border border-indigo-300 px-4 py-3 text-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-400"
            >
              {collections.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.metadata?.filename || col.name}
                </option>
              ))}
            </select>
          </div>

          <textarea
            rows={5}
            placeholder="Type your question here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full p-4 rounded-xl border border-indigo-300 resize-none text-gray-900 text-lg focus:outline-none focus:ring-2 focus:ring-indigo-400 transition bg-indigo-50"
          />

          <button
            onClick={handleAsk}
            disabled={loading}
            className={`flex items-center justify-center gap-2 w-full py-4 rounded-xl font-semibold shadow-md text-lg transition ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-green-600 hover:bg-green-700 text-white"
            }`}
          >
            {loading ? "⏳ Thinking..." : "🤖 Ask Now"}
          </button>
        </section>

        <section className="space-y-6">
          <h3 className="text-xl font-semibold text-indigo-700">🧠 Answer:</h3>
          <div className="p-5 rounded-xl border border-indigo-300 bg-white shadow-sm text-gray-900 min-h-[120px] whitespace-pre-wrap select-text">
            {answer || "No answer yet. Ask a question above!"}
          </div>

          {context && (
            <>
              <h4 className="text-md font-medium text-indigo-600">
                📚 Context:
              </h4>
              <div className="p-4 rounded-xl border border-indigo-200 bg-indigo-50 text-gray-700 whitespace-pre-wrap select-text">
                {context}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
