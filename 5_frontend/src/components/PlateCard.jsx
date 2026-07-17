const PLATE_TYPE_LABELS = {
  one_row: "Biển 1 hàng",
  two_row: "Biển 2 hàng",
};

function getConfidenceColor(conf) {
  if (conf >= 90) return "text-green-400";
  if (conf >= 70) return "text-yellow-400";
  return "text-red-400";
}

export default function PlateCard({ result }) {
  if (!result) return null;

  const conf = Math.round((result.confidence ?? 0) * 100);
  const confColor = getConfidenceColor(conf);
  const plateTypeLabel = PLATE_TYPE_LABELS[result.plate_type] || "Không xác định";
  const bboxText = Array.isArray(result.bbox) ? result.bbox.join(", ") : "N/A";

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden flex gap-4 p-4 items-center">
      {/* Ảnh crop biển số */}
      {result.image_crop && (
        <img
          src={`data:image/jpeg;base64,${result.image_crop}`}
          alt="Biển số"
          className="h-12 rounded object-contain bg-black border border-gray-600 flex-shrink-0"
        />
      )}

      {/* Thông tin */}
      <div className="flex-1 min-w-0">
        <div className="text-2xl font-bold tracking-widest text-white font-mono truncate">
          {result.plate_text || "Không nhận được"}
        </div>
        <div className="flex flex-wrap gap-3 mt-1 text-xs text-gray-400">
          <span className={`font-semibold ${confColor}`}>Độ tin cậy: {conf}%</span>
          <span>·</span>
          <span>{plateTypeLabel}</span>
          <span>·</span>
          <span>Bbox: [{bboxText}]</span>
        </div>
      </div>
    </div>
  );
}