import { useState, useRef } from "react";
import { predictVideo } from "../services/api";
import PlateCard from "../components/PlateCard";

// ---------------------------------------------------------------------
// Khoảng cách Levenshtein — đo "độ khác nhau" giữa 2 chuỗi (số lượt
// thêm/xóa/sửa ký tự tối thiểu để biến chuỗi này thành chuỗi kia).
// Dùng để phát hiện các lần đọc biển số GẦN GIỐNG NHAU qua nhiều frame
// (VD "30C16666" và "30C166N6" chỉ lệch 1 ký tự) -> khả năng cao là
// CÙNG 1 biển số thật, chỉ là model đọc sai lệch ở vài frame.
// ---------------------------------------------------------------------
function levenshtein(a, b) {
  const dp = Array.from({ length: a.length + 1 }, (_, i) =>
    Array(b.length + 1).fill(0).map((_, j) => (i === 0 ? j : 0))
  );
  for (let i = 0; i <= a.length; i++) dp[i][0] = i;
  for (let j = 0; j <= b.length; j++) dp[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[a.length][b.length];
}

// Ngưỡng chấp nhận "giống nhau": cho phép lệch tối đa ~25% độ dài
// chuỗi (tối thiểu 1, tối đa 2 ký tự) — đủ để gộp các lỗi đọc nhầm
// nhỏ lẻ, nhưng không gộp nhầm 2 biển số thật sự khác nhau.
function isSimilar(a, b) {
  if (Math.abs(a.length - b.length) > 2) return false;
  const maxLen = Math.max(a.length, b.length);
  const threshold = Math.min(2, Math.max(1, Math.round(maxLen * 0.25)));
  return levenshtein(a, b) <= threshold;
}

// Gộp danh sách detections (mỗi cái có plate_text, confidence, ...)
// thành các cụm biển số giống nhau, rồi BẦU CHỌN (voting) chuỗi text
// xuất hiện nhiều lần nhất trong mỗi cụm làm kết quả đại diện.
function clusterAndVote(detections) {
  const clusters = []; // [{ texts: {text: count}, items: [...] }]

  for (const det of detections) {
    const text = det.plate_text;
    let cluster = clusters.find((c) => isSimilar(text, c.representative));

    if (cluster) {
      cluster.items.push(det);
      cluster.texts[text] = (cluster.texts[text] || 0) + 1;
      // Cập nhật đại diện = text xuất hiện nhiều nhất trong cụm tới giờ
      cluster.representative = Object.entries(cluster.texts)
        .sort((a, b) => b[1] - a[1])[0][0];
    } else {
      clusters.push({
        representative: text,
        texts: { [text]: 1 },
        items: [det],
      });
    }
  }

  return clusters.map((c) => {
    const totalSeen = c.items.length;
    const bestText = Object.entries(c.texts).sort((a, b) => b[1] - a[1])[0];
    const [voteText, voteCount] = bestText;
    // Lấy item đại diện có confidence cao nhất TRONG SỐ các lần đọc ra
    // đúng chuỗi được bầu chọn (để hiển thị ảnh crop đẹp nhất)
    const bestItem = c.items
      .filter((it) => it.plate_text === voteText)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))[0];

    return {
      ...bestItem,
      plate_text: voteText,
      seen_count: totalSeen,
      vote_count: voteCount,
    };
  });
}

export default function VideoPage({ addHistory }) {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [interval, setInterval] = useState(10);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("video/")) return;
    setFile(f); setResults([]); setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResults([]);
    try {
      const data = await predictVideo(file, interval);
      setResults(data);
      const allDetections = data.flatMap(f => f.results);
      const clustered = clusterAndVote(allDetections);
      addHistory({ type: "video", filename: file.name, plates: clustered.map(c => c.plate_text) });
    } catch (e) {
      setError("Lỗi kết nối server. Hãy chắc chắn backend đang chạy.");
    } finally {
      setLoading(false);
    }
  };

  // Tổng hợp biển số duy nhất — gộp theo độ giống nhau + bầu chọn,
  // KHÔNG so khớp tuyệt đối (vì cùng 1 biển thật có thể bị đọc lệch
  // 1-2 ký tự khác nhau ở từng frame).
  const uniquePlates = clusterAndVote(results.flatMap(f => f.results))
    .sort((a, b) => b.seen_count - a.seen_count);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">Nhận dạng từ video</h2>

      {/* Upload */}
      <div className="bg-gray-900 rounded-2xl border border-gray-700 p-6 space-y-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => inputRef.current.click()}
            className="bg-gray-700 hover:bg-gray-600 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-all"
          >
            📂 Chọn video
          </button>
          {file && <span className="text-gray-300 text-sm">{file.name}</span>}
          <input ref={inputRef} type="file" accept="video/*" className="hidden"
            onChange={(e) => handleFile(e.target.files[0])} />
        </div>

        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400">Xử lý mỗi</label>
          <input
            type="number" min={1} max={60} value={interval}
            onChange={(e) => setInterval(Number(e.target.value))}
            className="w-20 bg-gray-800 border border-gray-600 text-white text-center rounded-lg px-2 py-1.5 text-sm"
          />
          <label className="text-sm text-gray-400">frame một lần</label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
            text-white font-semibold px-8 py-3 rounded-xl transition-all"
        >
          {loading ? "⏳ Đang xử lý video..." : "🎬 Bắt đầu nhận dạng"}
        </button>

        {loading && (
          <div className="flex items-center gap-3 text-gray-400 text-sm">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            Video đang được xử lý, vui lòng chờ...
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-900/40 border border-red-700 text-red-300 rounded-xl p-4 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Kết quả tổng hợp */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-200">
              Kết quả — {results.length} frame · {uniquePlates.length} biển số duy nhất (đã gộp)
            </h3>
          </div>

          {/* Biển số duy nhất */}
          <div className="bg-gray-900 rounded-xl border border-gray-700 p-4 space-y-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider">Tất cả biển số phát hiện</p>
            {uniquePlates.length === 0 ? (
              <p className="text-gray-500 text-sm">Không tìm thấy biển số nào</p>
            ) : (
              <div className="space-y-2">
                {uniquePlates.map((r, i) => (
                  <div key={i} className="space-y-1">
                    <PlateCard result={r} />
                    <p className="text-xs text-gray-500 pl-1">
                      Xuất hiện {r.seen_count} lần · đọc ra "{r.plate_text}" {r.vote_count}/{r.seen_count} lần
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Chi tiết theo frame */}
          <details className="bg-gray-900 rounded-xl border border-gray-700 p-4">
            <summary className="cursor-pointer text-sm text-gray-400 font-medium select-none">
              Chi tiết theo từng frame
            </summary>
            <div className="mt-4 space-y-4 max-h-96 overflow-y-auto pr-2">
              {results.filter(f => f.results.length > 0).map((frame) => (
                <div key={frame.frame_index} className="border-l-2 border-gray-700 pl-4">
                  <p className="text-xs text-gray-500 mb-2">
                    Frame #{frame.frame_index} — {(frame.timestamp_ms / 1000).toFixed(2)}s
                  </p>
                  {frame.results.map((r, i) => <PlateCard key={i} result={r} />)}
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}