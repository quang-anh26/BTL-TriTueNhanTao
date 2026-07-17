export default function HistoryPage({ history }) {
  const exportCSV = () => {
    const rows = [["Thời gian", "Loại", "File", "Biển số"]];
    history.forEach((h) => {
      h.plates.forEach((p) => rows.push([h.timestamp, h.type, h.filename, p]));
    });
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "lich_su_bien_so.csv"; a.click();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-200">Lịch sử nhận dạng</h2>
        {history.length > 0 && (
          <button
            onClick={exportCSV}
            className="bg-green-700 hover:bg-green-600 text-white text-sm font-medium px-4 py-2 rounded-xl transition-all"
          >
            ⬇️ Xuất CSV
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="bg-gray-900 rounded-2xl border border-gray-700 p-16 text-center text-gray-500">
          <div className="text-5xl mb-4">📋</div>
          <p>Chưa có lịch sử nào</p>
          <p className="text-sm mt-1">Hãy thử nhận dạng ảnh hoặc video trước</p>
        </div>
      ) : (
        <div className="space-y-3">
          {history.map((entry, i) => (
            <div key={i} className="bg-gray-900 rounded-xl border border-gray-700 p-4 flex gap-4 items-start">
              <div className="text-2xl">{entry.type === "image" ? "📷" : "🎬"}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-white truncate">{entry.filename}</span>
                  <span className="text-xs text-gray-500">{entry.timestamp}</span>
                </div>
                <div className="mt-2 flex gap-2 flex-wrap">
                  {entry.plates.length === 0 ? (
                    <span className="text-xs text-gray-500">Không tìm thấy biển số</span>
                  ) : (
                    entry.plates.map((p, j) => (
                      <span key={j} className="bg-gray-800 border border-gray-600 text-white font-mono font-bold text-sm px-3 py-1 rounded-lg">
                        {p}
                      </span>
                    ))
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
