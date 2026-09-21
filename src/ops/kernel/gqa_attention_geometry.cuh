#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

// Qwen3.5-9B shares the 35B's query-head count and the 27B's KV-head count, so
// neither existing geometry describes it and a dispatch on query heads alone
// silently sends it to <16, 2>. Splits follow the 35B: the grid is sized by
// query heads, and sixteen of them need the wider split to fill the device.
using Gqa9Geometry = GqaGeometry<16, 4, 2>;

} // namespace ninfer::ops
