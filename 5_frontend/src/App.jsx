import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import VideoPage from "./pages/VideoPage";
import HistoryPage from "./pages/HistoryPage";

const TABS = [
  { id: "image", label: "📷 Ảnh", icon: "📷" },
  { id: "video", label: "🎬 Video", icon: "🎬" },
  { id: "history", label: "📋 Lịch sử", icon: "📋" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("image");
  const [history, setHistory] = useState([]);

  const addHistory = (entry) => {
    setHistory((prev) => [{ ...entry, timestamp: new Date().toLocaleString("vi-VN") }, ...prev]);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 shadow-lg">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-4">
          <div className="text-3xl">🚗</div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              Nhận Dạng Biển Số Xe Việt Nam
            </h1>
            <p className="text-xs text-gray-400">Vietnamese License Plate Recognition · YOLOv8 + CRNN</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="max-w-6xl mx-auto px-6 flex gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-3 text-sm font-medium transition-all border-b-2 ${
                activeTab === tab.id
                  ? "border-blue-500 text-blue-400"
                  : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {activeTab === "image"   && <UploadPage  addHistory={addHistory} />}
        {activeTab === "video"   && <VideoPage   addHistory={addHistory} />}
        {activeTab === "history" && <HistoryPage history={history} />}
      </main>
    </div>
  );
}
