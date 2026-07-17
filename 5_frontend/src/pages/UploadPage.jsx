import { useState, useRef } from "react";
import { predictImage } from "../services/api";
import PlateCard from "../components/PlateCard";

export default function UploadPage({ addHistory }) {
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
      addHistory({ type: "image", filename: file.name, plates: data.results.map(r => r.plate_text) });
    } catch (e) {
      setError("Lỗi kết nối server. Hãy chắc chắn backend đang chạy.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">Nhận dạng từ ảnh</h2>

      {/* Drop Zone */}
      {!preview && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
          className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all
            ${dragging ? "border-blue-400 bg-blue-900/20" : "border-gray-700 hover:border-gray-500 hover:bg-gray-900"}`}
        >
          <div className="text-5xl mb-4">📸</div>
          <p className="text-gray-300 font-medium">Kéo thả ảnh vào đây</p>
          <p className="text-gray-500 text-sm mt-1">hoặc click để chọn file (JPG, PNG)</p>
          <input ref={inputRef} type="file" accept="image/*" className="hidden"
            onChange={(e) => handleFile(e.target.files[0])} />
        </div>
      )}

      {/* Preview + Result */}
      {preview && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Ảnh gốc / đã annotate */}
          <div className="space-y-3">
            <p className="text-sm text-gray-400 font-medium">
              {result ? "Kết quả nhận dạng" : "Ảnh đã chọn"}
            </p>
            <div className="rounded-xl overflow-hidden border border-gray-700 bg-black">
              <img
                src={result ? `data:image/jpeg;base64,${result.annotated_image}` : preview}
                alt="preview"
                className="w-full object-contain max-h-80"
              />
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
                  text-white font-semibold py-3 rounded-xl transition-all"
              >
                {loading ? "⏳ Đang xử lý..." : "🔍 Nhận dạng"}
              </button>
              <button
                onClick={handleReset}
                className="px-5 py-3 rounded-xl border border-gray-600 text-gray-300 hover:bg-gray-800 transition-all"
              >
                🔄 Làm lại
              </button>
            </div>
          </div>

          {/* Kết quả */}
          <div className="space-y-3">
            <p className="text-sm text-gray-400 font-medium">Biển số phát hiện được</p>

            {error && (
              <div className="bg-red-900/40 border border-red-700 text-red-300 rounded-xl p-4 text-sm">
                ⚠️ {error}
              </div>
            )}

            {result && (
              <>
                <div className="flex gap-4 text-sm text-gray-400">
                  <span>Tìm thấy <span className="text-white font-bold">{result.total_plates}</span> biển số</span>
                  <span>·</span>
                  <span>Thời gian: <span className="text-white font-bold">{result.inference_ms}ms</span></span>
                </div>

                {result.results.length === 0 ? (
                  <div className="bg-gray-800 rounded-xl p-8 text-center text-gray-500">
                    Không tìm thấy biển số nào
                  </div>
                ) : (
                  <div className="space-y-3">
                    {result.results.map((r, i) => (
                      <PlateCard key={i} result={r} />
                    ))}
                  </div>
                )}
              </>
            )}

            {!result && !error && (
              <div className="bg-gray-800/50 rounded-xl p-8 text-center text-gray-500 border border-gray-700">
                Nhấn "Nhận dạng" để bắt đầu
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
