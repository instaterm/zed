// =====================================================================
// FORK NOTE — instaterm/zed only (branch: 0.232.x-shape-line-options)
// =====================================================================
// This file carries an additive `ShapeLineOptions` API for grid-aligned
// glyph snapping (used by instaterm's terminal renderer to keep
// ligatures, emoji clusters, and multi-byte UTF-8 graphemes correctly
// aligned to a forced cell grid). The upstream gpui has no equivalent.
//
// What was added (look for these on rebase):
//   - `SnapMode` / `SnapModeRef` / `ShapeLineOptions` / `ShapeLineOptions::normalized()`
//     near the top of this file. Pure additions, low conflict risk.
//   - `snap_glyph_positions()` helper. The previous inline snap loops
//     in `LineLayoutCache::layout_line` and `..._by_hash` were replaced
//     with calls into this helper. If upstream rewrites either snap
//     loop, replay the change into the helper rather than reverting.
//   - `_with` siblings on `LineLayoutCache`: `layout_line_with`,
//     `try_layout_line_with_by_hash`, `layout_line_with_by_hash`. The
//     legacy `layout_line` delegates to `layout_line_with` with default
//     options. The legacy `*_by_hash` cache wrappers were deleted —
//     `WindowTextSystem::*_by_hash` calls `*_with_by_hash` directly.
//
// High-conflict areas on upstream rebase:
//   1. `CacheKey`/`CacheKeyRef`/`HashedCacheKey`/`HashedCacheKeyRef` —
//      we added `snap_tolerance: Pixels` and `snap_mode: SnapMode`
//      (owned) / `SnapModeRef<'a>` (borrowed). Every struct literal
//      that builds these keys (in `layout_wrapped_line`,
//      `layout_line_with`, `try_layout_line_with_by_hash`,
//      `layout_line_with_by_hash`, `as_cache_key_ref`, `to_ref`) needs
//      the new fields. If upstream adds/removes a field, mirror into
//      ALL of those sites.
//   2. The `Hash`/`PartialEq` impls for `HashedCacheKey` /
//      `HashedCacheKeyRef` were converted from inline field-by-field
//      to delegate via `to_ref()`. Keep this shape — it's why the
//      borrowed/owned forms produce identical hashes.
//
// Cache-key invariants (preserve these on rebase):
//   - `SnapModeRef` MUST stay `Copy` so `CacheKeyRef` stays `Copy`
//     (probe paths run per-row per-frame in the terminal renderer).
//     `SnapMode::ByteToColumn` holds an `Arc<[u32]>` — cheap to clone,
//     hashes by slice contents not pointer identity.
//   - `ShapeLineOptions::normalized()` MUST be called before building
//     a cache key, so `force_width = None` collapses snap fields to
//     defaults (otherwise cache hits fragment on irrelevant inputs).
//
// Public API surface (deliberately narrow — keep it that way):
//   - Public: `SnapMode`, `ShapeLineOptions` (this file);
//     `WindowTextSystem::shape_line_with` (text_system.rs).
//   - `pub(crate)` only: `SnapModeRef`, all other `_with` methods on
//     `WindowTextSystem` and `LineLayoutCache`. instaterm only uses
//     `shape_line_with`.
//
// =====================================================================

use crate::{FontId, GlyphId, Pixels, PlatformTextSystem, Point, SharedString, Size, point, px};
use collections::FxHashMap;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
};

use super::LineWrapper;

/// How glyphs snap onto the forced grid when `force_width` is set.
#[derive(Clone, Debug)]
pub enum SnapMode {
    /// Snap glyph N to `N × force_width`. Correct when 1 glyph = 1 column
    /// (no ligatures or contextual substitutions that fuse multiple input
    /// positions into one glyph). This is the legacy behaviour.
    GlyphIndex,
    /// Snap a glyph whose source starts at byte B to
    /// `byte_to_column[B] × force_width`. The table must have length
    /// `text.len() + 1`; the last entry is the total column count, used
    /// to compute `layout.width = byte_to_column[text.len()] × force_width`.
    ///
    /// "Column" here means one `force_width` stride — gpui has no opinion
    /// on what one stride represents to the caller. Two valid usages:
    ///
    /// - **Uniform grid:** `force_width` = per-cell pixel width; the table
    ///   tracks per-cell ordinals (`[0, 1, 1, 2, ..]` for `"a你b"` where
    ///   `你` spans columns 1–2 → byte_to_column maps `你`'s 3 bytes to 1).
    /// - **Mixed-width grid:** caller splits text into per-stride spans
    ///   (one span per width class) and treats each span's stride as one
    ///   column. A terminal renderer with width-2 CJK cells uses
    ///   `force_width = 2 × base_cell_width` for its CJK spans and tracks
    ///   stride ordinals (sentinel `n` for `n` width-2 cells, not `2n`).
    ///
    /// Use this whenever source bytes and output columns are not in 1:1
    /// correspondence — ligatures, emoji clusters, multi-byte UTF-8
    /// graphemes.
    ByteToColumn {
        /// Byte offset → column ordinal. See variant docs.
        byte_to_column: Arc<[u32]>,
    },
}

impl SnapMode {
    fn as_ref(&self) -> SnapModeRef<'_> {
        match self {
            SnapMode::GlyphIndex => SnapModeRef::GlyphIndex,
            SnapMode::ByteToColumn { byte_to_column } => {
                SnapModeRef::ByteToColumn { byte_to_column }
            }
        }
    }
}

impl PartialEq for SnapMode {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for SnapMode {}

impl Hash for SnapMode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

/// Borrowed sibling of [`SnapMode`], used in cache probe keys to keep them `Copy`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SnapModeRef<'a> {
    GlyphIndex,
    ByteToColumn { byte_to_column: &'a [u32] },
}

impl PartialEq for SnapModeRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SnapModeRef::GlyphIndex, SnapModeRef::GlyphIndex) => true,
            (
                SnapModeRef::ByteToColumn { byte_to_column: a },
                SnapModeRef::ByteToColumn { byte_to_column: b },
            ) => a == b,
            _ => false,
        }
    }
}

impl Eq for SnapModeRef<'_> {}

impl Hash for SnapModeRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            SnapModeRef::GlyphIndex => {
                0u8.hash(state);
            }
            SnapModeRef::ByteToColumn { byte_to_column } => {
                1u8.hash(state);
                byte_to_column.hash(state);
            }
        }
    }
}

/// Options controlling glyph snapping when shaping a line with a forced column grid.
///
/// The default value matches the legacy `shape_line(.., force_width = None)` behaviour:
/// no snapping, 1px tolerance, `GlyphIndex` snap mode.
///
/// Note: when `force_width` is `None`, `snap_tolerance` and `snap_mode` have no effect
/// on the resulting layout. `ShapeLineOptions` is normalized at each entry point so
/// non-default values in that case collapse to defaults for cache-key purposes —
/// cache hits therefore don't fragment on those irrelevant fields.
#[derive(Clone, Debug)]
pub struct ShapeLineOptions {
    /// If set, glyphs are snapped onto a grid of this width. `None` means no snap.
    pub force_width: Option<Pixels>,
    /// Glyphs only snap if their natural position differs from the target by more
    /// than this tolerance. Legacy behaviour is `px(1.)`.
    pub snap_tolerance: Pixels,
    /// How the snap target column is derived. See [`SnapMode`].
    pub snap_mode: SnapMode,
}

impl Default for ShapeLineOptions {
    fn default() -> Self {
        Self {
            force_width: None,
            snap_tolerance: px(1.),
            snap_mode: SnapMode::GlyphIndex,
        }
    }
}

impl ShapeLineOptions {
    /// Normalize to a canonical form for cache keying. When `force_width` is `None`
    /// the snap loop is skipped entirely, so `snap_tolerance` and `snap_mode`
    /// cannot influence the resulting layout — collapse them to defaults so cache
    /// hits are maximised.
    pub(crate) fn normalized(mut self) -> Self {
        if self.force_width.is_none() {
            self.snap_tolerance = px(1.);
            self.snap_mode = SnapMode::GlyphIndex;
        }
        self
    }
}

fn snap_glyph_positions(
    layout: &mut LineLayout,
    force_width: Pixels,
    snap_tolerance: Pixels,
    snap_mode: &SnapMode,
    text_len: usize,
) {
    if let SnapMode::ByteToColumn { byte_to_column } = snap_mode {
        debug_assert_eq!(
            byte_to_column.len(),
            text_len + 1,
            "byte_to_column must have exactly text.len() + 1 entries (one per byte plus a sentinel)",
        );
        debug_assert!(
            byte_to_column.windows(2).all(|w| w[0] <= w[1]),
            "byte_to_column must be monotonically non-decreasing",
        );
    }

    let mut glyph_pos: u32 = 0;
    for run in layout.runs.iter_mut() {
        for glyph in run.glyphs.iter_mut() {
            let target_x: Pixels = match snap_mode {
                SnapMode::GlyphIndex => glyph_pos as f32 * force_width,
                SnapMode::ByteToColumn { byte_to_column } => {
                    byte_to_column[glyph.index] as f32 * force_width
                }
            };
            if (glyph.position.x - target_x).abs() > snap_tolerance {
                glyph.position.x = target_x;
            }
            glyph_pos += 1;
        }
    }
    layout.width = match snap_mode {
        SnapMode::GlyphIndex => glyph_pos as f32 * force_width,
        SnapMode::ByteToColumn { byte_to_column } => byte_to_column[text_len] as f32 * force_width,
    };
}

/// A laid out and styled line of text
#[derive(Default, Debug)]
pub struct LineLayout {
    /// The font size for this line
    pub font_size: Pixels,
    /// The width of the line
    pub width: Pixels,
    /// The ascent of the line
    pub ascent: Pixels,
    /// The descent of the line
    pub descent: Pixels,
    /// The shaped runs that make up this line
    pub runs: Vec<ShapedRun>,
    /// The length of the line in utf-8 bytes
    pub len: usize,
}

/// A run of text that has been shaped .
#[derive(Debug, Clone)]
pub struct ShapedRun {
    /// The font id for this run
    pub font_id: FontId,
    /// The glyphs that make up this run
    pub glyphs: Vec<ShapedGlyph>,
}

/// A single glyph, ready to paint.
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    /// The ID for this glyph, as determined by the text system.
    pub id: GlyphId,

    /// The position of this glyph in its containing line.
    pub position: Point<Pixels>,

    /// The index of this glyph in the original text.
    pub index: usize,

    /// Whether this glyph is an emoji
    pub is_emoji: bool,
}

impl LineLayout {
    /// The index for the character at the given x coordinate
    pub fn index_for_x(&self, x: Pixels) -> Option<usize> {
        if x >= self.width {
            None
        } else {
            for run in self.runs.iter().rev() {
                for glyph in run.glyphs.iter().rev() {
                    if glyph.position.x <= x {
                        return Some(glyph.index);
                    }
                }
            }
            Some(0)
        }
    }

    /// closest_index_for_x returns the character boundary closest to the given x coordinate
    /// (e.g. to handle aligning up/down arrow keys)
    pub fn closest_index_for_x(&self, x: Pixels) -> usize {
        let mut prev_index = 0;
        let mut prev_x = px(0.);

        for run in self.runs.iter() {
            for glyph in run.glyphs.iter() {
                if glyph.position.x >= x {
                    if glyph.position.x - x < x - prev_x {
                        return glyph.index;
                    } else {
                        return prev_index;
                    }
                }
                prev_index = glyph.index;
                prev_x = glyph.position.x;
            }
        }

        if self.len == 1 {
            if x > self.width / 2. {
                return 1;
            } else {
                return 0;
            }
        }

        self.len
    }

    /// The x position of the character at the given index
    pub fn x_for_index(&self, index: usize) -> Pixels {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return glyph.position.x;
                }
            }
        }
        self.width
    }

    /// The corresponding Font at the given index
    pub fn font_id_for_index(&self, index: usize) -> Option<FontId> {
        for run in &self.runs {
            for glyph in &run.glyphs {
                if glyph.index >= index {
                    return Some(run.font_id);
                }
            }
        }
        None
    }

    fn compute_wrap_boundaries(
        &self,
        text: &str,
        wrap_width: Pixels,
        max_lines: Option<usize>,
    ) -> SmallVec<[WrapBoundary; 1]> {
        let mut boundaries = SmallVec::new();
        let mut first_non_whitespace_ix = None;
        let mut last_candidate_ix = None;
        let mut last_candidate_x = px(0.);
        let mut last_boundary = WrapBoundary {
            run_ix: 0,
            glyph_ix: 0,
        };
        let mut last_boundary_x = px(0.);
        let mut prev_ch = '\0';
        let mut glyphs = self
            .runs
            .iter()
            .enumerate()
            .flat_map(move |(run_ix, run)| {
                run.glyphs.iter().enumerate().map(move |(glyph_ix, glyph)| {
                    let character = text[glyph.index..].chars().next().unwrap();
                    (
                        WrapBoundary { run_ix, glyph_ix },
                        character,
                        glyph.position.x,
                    )
                })
            })
            .peekable();

        while let Some((boundary, ch, x)) = glyphs.next() {
            if ch == '\n' {
                continue;
            }

            // Here is very similar to `LineWrapper::wrap_line` to determine text wrapping,
            // but there are some differences, so we have to duplicate the code here.
            if LineWrapper::is_word_char(ch) {
                if prev_ch == ' ' && ch != ' ' && first_non_whitespace_ix.is_some() {
                    last_candidate_ix = Some(boundary);
                    last_candidate_x = x;
                }
            } else {
                if ch != ' ' && first_non_whitespace_ix.is_some() {
                    last_candidate_ix = Some(boundary);
                    last_candidate_x = x;
                }
            }

            if ch != ' ' && first_non_whitespace_ix.is_none() {
                first_non_whitespace_ix = Some(boundary);
            }

            let next_x = glyphs.peek().map_or(self.width, |(_, _, x)| *x);
            let width = next_x - last_boundary_x;

            if width > wrap_width && boundary > last_boundary {
                // When used line_clamp, we should limit the number of lines.
                if let Some(max_lines) = max_lines
                    && boundaries.len() >= max_lines - 1
                {
                    break;
                }

                if let Some(last_candidate_ix) = last_candidate_ix.take() {
                    last_boundary = last_candidate_ix;
                    last_boundary_x = last_candidate_x;
                } else {
                    last_boundary = boundary;
                    last_boundary_x = x;
                }
                boundaries.push(last_boundary);
            }
            prev_ch = ch;
        }

        boundaries
    }
}

/// A line of text that has been wrapped to fit a given width
#[derive(Default, Debug)]
pub struct WrappedLineLayout {
    /// The line layout, pre-wrapping.
    pub unwrapped_layout: Arc<LineLayout>,

    /// The boundaries at which the line was wrapped
    pub wrap_boundaries: SmallVec<[WrapBoundary; 1]>,

    /// The width of the line, if it was wrapped
    pub wrap_width: Option<Pixels>,
}

/// A boundary at which a line was wrapped
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WrapBoundary {
    /// The index in the run just before the line was wrapped
    pub run_ix: usize,
    /// The index of the glyph just before the line was wrapped
    pub glyph_ix: usize,
}

impl WrappedLineLayout {
    /// The length of the underlying text, in utf8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.unwrapped_layout.len
    }

    /// The width of this line, in pixels, whether or not it was wrapped.
    pub fn width(&self) -> Pixels {
        self.wrap_width
            .unwrap_or(Pixels::MAX)
            .min(self.unwrapped_layout.width)
    }

    /// The size of the whole wrapped text, for the given line_height.
    /// can span multiple lines if there are multiple wrap boundaries.
    pub fn size(&self, line_height: Pixels) -> Size<Pixels> {
        Size {
            width: self.width(),
            height: line_height * (self.wrap_boundaries.len() + 1),
        }
    }

    /// The ascent of a line in this layout
    pub fn ascent(&self) -> Pixels {
        self.unwrapped_layout.ascent
    }

    /// The descent of a line in this layout
    pub fn descent(&self) -> Pixels {
        self.unwrapped_layout.descent
    }

    /// The wrap boundaries in this layout
    pub fn wrap_boundaries(&self) -> &[WrapBoundary] {
        &self.wrap_boundaries
    }

    /// The font size of this layout
    pub fn font_size(&self) -> Pixels {
        self.unwrapped_layout.font_size
    }

    /// The runs in this layout, sans wrapping
    pub fn runs(&self) -> &[ShapedRun] {
        &self.unwrapped_layout.runs
    }

    /// The index corresponding to a given position in this layout for the given line height.
    ///
    /// See also [`Self::closest_index_for_position`].
    pub fn index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, false)
    }

    /// The closest index to a given position in this layout for the given line height.
    ///
    /// Closest means the character boundary closest to the given position.
    ///
    /// See also [`LineLayout::closest_index_for_x`].
    pub fn closest_index_for_position(
        &self,
        position: Point<Pixels>,
        line_height: Pixels,
    ) -> Result<usize, usize> {
        self._index_for_position(position, line_height, true)
    }

    fn _index_for_position(
        &self,
        mut position: Point<Pixels>,
        line_height: Pixels,
        closest: bool,
    ) -> Result<usize, usize> {
        let wrapped_line_ix = (position.y / line_height) as usize;

        let wrapped_line_start_index;
        let wrapped_line_start_x;
        if wrapped_line_ix > 0 {
            let Some(line_start_boundary) = self.wrap_boundaries.get(wrapped_line_ix - 1) else {
                return Err(0);
            };
            let run = &self.unwrapped_layout.runs[line_start_boundary.run_ix];
            let glyph = &run.glyphs[line_start_boundary.glyph_ix];
            wrapped_line_start_index = glyph.index;
            wrapped_line_start_x = glyph.position.x;
        } else {
            wrapped_line_start_index = 0;
            wrapped_line_start_x = Pixels::ZERO;
        };

        let wrapped_line_end_index;
        let wrapped_line_end_x;
        if wrapped_line_ix < self.wrap_boundaries.len() {
            let next_wrap_boundary_ix = wrapped_line_ix;
            let next_wrap_boundary = self.wrap_boundaries[next_wrap_boundary_ix];
            let run = &self.unwrapped_layout.runs[next_wrap_boundary.run_ix];
            let glyph = &run.glyphs[next_wrap_boundary.glyph_ix];
            wrapped_line_end_index = glyph.index;
            wrapped_line_end_x = glyph.position.x;
        } else {
            wrapped_line_end_index = self.unwrapped_layout.len;
            wrapped_line_end_x = self.unwrapped_layout.width;
        };

        let mut position_in_unwrapped_line = position;
        position_in_unwrapped_line.x += wrapped_line_start_x;
        if position_in_unwrapped_line.x < wrapped_line_start_x {
            Err(wrapped_line_start_index)
        } else if position_in_unwrapped_line.x >= wrapped_line_end_x {
            Err(wrapped_line_end_index)
        } else {
            if closest {
                Ok(self
                    .unwrapped_layout
                    .closest_index_for_x(position_in_unwrapped_line.x))
            } else {
                Ok(self
                    .unwrapped_layout
                    .index_for_x(position_in_unwrapped_line.x)
                    .unwrap())
            }
        }
    }

    /// Returns the pixel position for the given byte index.
    pub fn position_for_index(&self, index: usize, line_height: Pixels) -> Option<Point<Pixels>> {
        let mut line_start_ix = 0;
        let mut line_end_indices = self
            .wrap_boundaries
            .iter()
            .map(|wrap_boundary| {
                let run = &self.unwrapped_layout.runs[wrap_boundary.run_ix];
                let glyph = &run.glyphs[wrap_boundary.glyph_ix];
                glyph.index
            })
            .chain([self.len()])
            .enumerate();
        for (ix, line_end_ix) in line_end_indices {
            let line_y = ix as f32 * line_height;
            if index < line_start_ix {
                break;
            } else if index > line_end_ix {
                line_start_ix = line_end_ix;
                continue;
            } else {
                let line_start_x = self.unwrapped_layout.x_for_index(line_start_ix);
                let x = self.unwrapped_layout.x_for_index(index) - line_start_x;
                return Some(point(x, line_y));
            }
        }

        None
    }
}

pub(crate) struct LineLayoutCache {
    previous_frame: Mutex<FrameCache>,
    current_frame: RwLock<FrameCache>,
    platform_text_system: Arc<dyn PlatformTextSystem>,
}

#[derive(Default)]
struct FrameCache {
    lines: FxHashMap<Arc<CacheKey>, Arc<LineLayout>>,
    wrapped_lines: FxHashMap<Arc<CacheKey>, Arc<WrappedLineLayout>>,
    used_lines: Vec<Arc<CacheKey>>,
    used_wrapped_lines: Vec<Arc<CacheKey>>,

    // Content-addressable caches keyed by caller-provided text hash + layout params.
    // These allow cache hits without materializing a contiguous `SharedString`.
    //
    // IMPORTANT: To support allocation-free lookups, we store these maps using a key type
    // (`HashedCacheKeyRef`) that can be computed without building a contiguous `&str`/`SharedString`.
    // On miss, we allocate once and store under an owned `HashedCacheKey`.
    lines_by_hash: FxHashMap<Arc<HashedCacheKey>, Arc<LineLayout>>,
    wrapped_lines_by_hash: FxHashMap<Arc<HashedCacheKey>, Arc<WrappedLineLayout>>,
    used_lines_by_hash: Vec<Arc<HashedCacheKey>>,
    used_wrapped_lines_by_hash: Vec<Arc<HashedCacheKey>>,
}

#[derive(Clone, Default)]
pub(crate) struct LineLayoutIndex {
    lines_index: usize,
    wrapped_lines_index: usize,
    lines_by_hash_index: usize,
    wrapped_lines_by_hash_index: usize,
}

impl LineLayoutCache {
    pub fn new(platform_text_system: Arc<dyn PlatformTextSystem>) -> Self {
        Self {
            previous_frame: Mutex::default(),
            current_frame: RwLock::default(),
            platform_text_system,
        }
    }

    pub fn layout_index(&self) -> LineLayoutIndex {
        let frame = self.current_frame.read();
        LineLayoutIndex {
            lines_index: frame.used_lines.len(),
            wrapped_lines_index: frame.used_wrapped_lines.len(),
            lines_by_hash_index: frame.used_lines_by_hash.len(),
            wrapped_lines_by_hash_index: frame.used_wrapped_lines_by_hash.len(),
        }
    }

    pub fn reuse_layouts(&self, range: Range<LineLayoutIndex>) {
        let mut previous_frame = &mut *self.previous_frame.lock();
        let mut current_frame = &mut *self.current_frame.write();

        for key in &previous_frame.used_lines[range.start.lines_index..range.end.lines_index] {
            if let Some((key, line)) = previous_frame.lines.remove_entry(key) {
                current_frame.lines.insert(key, line);
            }
            current_frame.used_lines.push(key.clone());
        }

        for key in &previous_frame.used_wrapped_lines
            [range.start.wrapped_lines_index..range.end.wrapped_lines_index]
        {
            if let Some((key, line)) = previous_frame.wrapped_lines.remove_entry(key) {
                current_frame.wrapped_lines.insert(key, line);
            }
            current_frame.used_wrapped_lines.push(key.clone());
        }

        for key in &previous_frame.used_lines_by_hash
            [range.start.lines_by_hash_index..range.end.lines_by_hash_index]
        {
            if let Some((key, line)) = previous_frame.lines_by_hash.remove_entry(key) {
                current_frame.lines_by_hash.insert(key, line);
            }
            current_frame.used_lines_by_hash.push(key.clone());
        }

        for key in &previous_frame.used_wrapped_lines_by_hash
            [range.start.wrapped_lines_by_hash_index..range.end.wrapped_lines_by_hash_index]
        {
            if let Some((key, line)) = previous_frame.wrapped_lines_by_hash.remove_entry(key) {
                current_frame.wrapped_lines_by_hash.insert(key, line);
            }
            current_frame.used_wrapped_lines_by_hash.push(key.clone());
        }
    }

    pub fn truncate_layouts(&self, index: LineLayoutIndex) {
        let mut current_frame = &mut *self.current_frame.write();
        current_frame.used_lines.truncate(index.lines_index);
        current_frame
            .used_wrapped_lines
            .truncate(index.wrapped_lines_index);
        current_frame
            .used_lines_by_hash
            .truncate(index.lines_by_hash_index);
        current_frame
            .used_wrapped_lines_by_hash
            .truncate(index.wrapped_lines_by_hash_index);
    }

    pub fn finish_frame(&self) {
        let mut prev_frame = self.previous_frame.lock();
        let mut curr_frame = self.current_frame.write();
        std::mem::swap(&mut *prev_frame, &mut *curr_frame);
        curr_frame.lines.clear();
        curr_frame.wrapped_lines.clear();
        curr_frame.used_lines.clear();
        curr_frame.used_wrapped_lines.clear();

        curr_frame.lines_by_hash.clear();
        curr_frame.wrapped_lines_by_hash.clear();
        curr_frame.used_lines_by_hash.clear();
        curr_frame.used_wrapped_lines_by_hash.clear();
    }

    pub fn layout_wrapped_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        wrap_width: Option<Pixels>,
        max_lines: Option<usize>,
    ) -> Arc<WrappedLineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width,
            force_width: None,
            snap_tolerance: px(1.),
            snap_mode: SnapModeRef::GlyphIndex,
        } as &dyn AsCacheKeyRef;

        let current_frame = self.current_frame.upgradable_read();
        if let Some(layout) = current_frame.wrapped_lines.get(key) {
            return layout.clone();
        }

        let previous_frame_entry = self.previous_frame.lock().wrapped_lines.remove_entry(key);
        if let Some((key, layout)) = previous_frame_entry {
            let mut current_frame = RwLockUpgradableReadGuard::upgrade(current_frame);
            current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            current_frame.used_wrapped_lines.push(key);
            layout
        } else {
            drop(current_frame);
            let text = SharedString::from(text);
            let unwrapped_layout = self.layout_line::<&SharedString>(&text, font_size, runs, None);
            let wrap_boundaries = if let Some(wrap_width) = wrap_width {
                unwrapped_layout.compute_wrap_boundaries(text.as_ref(), wrap_width, max_lines)
            } else {
                SmallVec::new()
            };
            let layout = Arc::new(WrappedLineLayout {
                unwrapped_layout,
                wrap_boundaries,
                wrap_width,
            });
            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width,
                force_width: None,
                snap_tolerance: px(1.),
                snap_mode: SnapMode::GlyphIndex,
            });

            let mut current_frame = self.current_frame.write();
            current_frame
                .wrapped_lines
                .insert(key.clone(), layout.clone());
            current_frame.used_wrapped_lines.push(key);

            layout
        }
    }

    pub fn layout_line<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        force_width: Option<Pixels>,
    ) -> Arc<LineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        self.layout_line_with(
            text,
            font_size,
            runs,
            ShapeLineOptions {
                force_width,
                ..Default::default()
            },
        )
    }

    pub fn layout_line_with<Text>(
        &self,
        text: Text,
        font_size: Pixels,
        runs: &[FontRun],
        opts: ShapeLineOptions,
    ) -> Arc<LineLayout>
    where
        Text: AsRef<str>,
        SharedString: From<Text>,
    {
        let opts = opts.normalized();
        let key = &CacheKeyRef {
            text: text.as_ref(),
            font_size,
            runs,
            wrap_width: None,
            force_width: opts.force_width,
            snap_tolerance: opts.snap_tolerance,
            snap_mode: opts.snap_mode.as_ref(),
        } as &dyn AsCacheKeyRef;

        let current_frame = self.current_frame.upgradable_read();
        if let Some(layout) = current_frame.lines.get(key) {
            return layout.clone();
        }

        let mut current_frame = RwLockUpgradableReadGuard::upgrade(current_frame);
        if let Some((key, layout)) = self.previous_frame.lock().lines.remove_entry(key) {
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        } else {
            let text = SharedString::from(text);
            let mut layout = self
                .platform_text_system
                .layout_line(&text, font_size, runs);

            if let Some(force_width) = opts.force_width {
                snap_glyph_positions(
                    &mut layout,
                    force_width,
                    opts.snap_tolerance,
                    &opts.snap_mode,
                    text.len(),
                );
            }

            let key = Arc::new(CacheKey {
                text,
                font_size,
                runs: SmallVec::from(runs),
                wrap_width: None,
                force_width: opts.force_width,
                snap_tolerance: opts.snap_tolerance,
                snap_mode: opts.snap_mode,
            });
            let layout = Arc::new(layout);
            current_frame.lines.insert(key.clone(), layout.clone());
            current_frame.used_lines.push(key);
            layout
        }
    }

    /// Try to retrieve a previously-shaped line layout using a caller-provided content hash.
    ///
    /// This is a *non-allocating* cache probe: it does not materialize any text. If the layout
    /// is not already cached in either the current frame or previous frame, returns `None`.
    ///
    /// Contract (caller enforced):
    /// - Same `text_hash` implies identical text content (collision risk accepted by caller).
    /// - `text_len` should be the UTF-8 byte length of the text (helps reduce accidental collisions).
    pub fn try_layout_line_with_by_hash(
        &self,
        text_hash: u64,
        text_len: usize,
        font_size: Pixels,
        runs: &[FontRun],
        opts: ShapeLineOptions,
    ) -> Option<Arc<LineLayout>> {
        let opts = opts.normalized();
        let snap_mode_ref = opts.snap_mode.as_ref();
        let key_ref = HashedCacheKeyRef {
            text_hash,
            text_len,
            font_size,
            runs,
            wrap_width: None,
            force_width: opts.force_width,
            snap_tolerance: opts.snap_tolerance,
            snap_mode: snap_mode_ref,
        };

        let current_frame = self.current_frame.read();
        if let Some((_, layout)) = current_frame
            .lines_by_hash
            .iter()
            .find(|(key, _)| key.to_ref() == key_ref)
        {
            return Some(layout.clone());
        }

        let previous_frame = self.previous_frame.lock();
        if let Some((_, layout)) = previous_frame
            .lines_by_hash
            .iter()
            .find(|(key, _)| key.to_ref() == key_ref)
        {
            return Some(layout.clone());
        }

        None
    }

    /// Layout a line of text using a caller-provided content hash as the cache key.
    ///
    /// This enables cache hits without materializing a contiguous `SharedString` for `text`.
    /// If the cache misses, `materialize_text` is invoked to produce the `SharedString` for shaping.
    ///
    /// Contract (caller enforced):
    /// - Same `text_hash` implies identical text content (collision risk accepted by caller).
    /// - `text_len` should be the UTF-8 byte length of the text (helps reduce accidental collisions).
    pub fn layout_line_with_by_hash(
        &self,
        text_hash: u64,
        text_len: usize,
        font_size: Pixels,
        runs: &[FontRun],
        opts: ShapeLineOptions,
        materialize_text: impl FnOnce() -> SharedString,
    ) -> Arc<LineLayout> {
        let opts = opts.normalized();
        let snap_mode_ref = opts.snap_mode.as_ref();
        let key_ref = HashedCacheKeyRef {
            text_hash,
            text_len,
            font_size,
            runs,
            wrap_width: None,
            force_width: opts.force_width,
            snap_tolerance: opts.snap_tolerance,
            snap_mode: snap_mode_ref,
        };

        // Fast path: already cached (no allocation).
        let current_frame = self.current_frame.upgradable_read();
        if let Some((_, layout)) = current_frame
            .lines_by_hash
            .iter()
            .find(|(key, _)| key.to_ref() == key_ref)
        {
            return layout.clone();
        }

        let mut current_frame = RwLockUpgradableReadGuard::upgrade(current_frame);

        // Try to reuse from previous frame without allocating; do a linear scan to find a matching key.
        // (We avoid `drain()` here because it would eagerly move all entries.)
        let mut previous_frame = self.previous_frame.lock();
        if let Some(existing_key) = previous_frame
            .used_lines_by_hash
            .iter()
            .find(|key| key.to_ref() == key_ref)
            .cloned()
        {
            if let Some((key, layout)) = previous_frame.lines_by_hash.remove_entry(&existing_key) {
                current_frame
                    .lines_by_hash
                    .insert(key.clone(), layout.clone());
                current_frame.used_lines_by_hash.push(key);
                return layout;
            }
        }

        let text = materialize_text();
        let mut layout = self
            .platform_text_system
            .layout_line(&text, font_size, runs);

        if let Some(force_width) = opts.force_width {
            snap_glyph_positions(
                &mut layout,
                force_width,
                opts.snap_tolerance,
                &opts.snap_mode,
                text.len(),
            );
        }

        let key = Arc::new(HashedCacheKey {
            text_hash,
            text_len,
            font_size,
            runs: SmallVec::from(runs),
            wrap_width: None,
            force_width: opts.force_width,
            snap_tolerance: opts.snap_tolerance,
            snap_mode: opts.snap_mode,
        });
        let layout = Arc::new(layout);
        current_frame
            .lines_by_hash
            .insert(key.clone(), layout.clone());
        current_frame.used_lines_by_hash.push(key);
        layout
    }
}

/// A run of text with a single font.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[expect(missing_docs)]
pub struct FontRun {
    pub len: usize,
    pub font_id: FontId,
}

trait AsCacheKeyRef {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_>;
}

#[derive(Clone, Debug, Eq)]
struct CacheKey {
    text: SharedString,
    font_size: Pixels,
    runs: SmallVec<[FontRun; 1]>,
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    snap_tolerance: Pixels,
    snap_mode: SnapMode,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct CacheKeyRef<'a> {
    text: &'a str,
    font_size: Pixels,
    runs: &'a [FontRun],
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    snap_tolerance: Pixels,
    snap_mode: SnapModeRef<'a>,
}

#[derive(Clone, Debug)]
struct HashedCacheKey {
    text_hash: u64,
    text_len: usize,
    font_size: Pixels,
    runs: SmallVec<[FontRun; 1]>,
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    snap_tolerance: Pixels,
    snap_mode: SnapMode,
}

#[derive(Copy, Clone)]
struct HashedCacheKeyRef<'a> {
    text_hash: u64,
    text_len: usize,
    font_size: Pixels,
    runs: &'a [FontRun],
    wrap_width: Option<Pixels>,
    force_width: Option<Pixels>,
    snap_tolerance: Pixels,
    snap_mode: SnapModeRef<'a>,
}

impl PartialEq for dyn AsCacheKeyRef + '_ {
    fn eq(&self, other: &dyn AsCacheKeyRef) -> bool {
        self.as_cache_key_ref() == other.as_cache_key_ref()
    }
}

impl HashedCacheKey {
    fn to_ref(&self) -> HashedCacheKeyRef<'_> {
        HashedCacheKeyRef {
            text_hash: self.text_hash,
            text_len: self.text_len,
            font_size: self.font_size,
            runs: self.runs.as_slice(),
            wrap_width: self.wrap_width,
            force_width: self.force_width,
            snap_tolerance: self.snap_tolerance,
            snap_mode: self.snap_mode.as_ref(),
        }
    }
}

impl PartialEq for HashedCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.to_ref() == other.to_ref()
    }
}

impl Eq for HashedCacheKey {}

impl Hash for HashedCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_ref().hash(state);
    }
}

impl PartialEq for HashedCacheKeyRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.text_hash == other.text_hash
            && self.text_len == other.text_len
            && self.font_size == other.font_size
            && self.runs == other.runs
            && self.wrap_width == other.wrap_width
            && self.force_width == other.force_width
            && self.snap_tolerance == other.snap_tolerance
            && self.snap_mode == other.snap_mode
    }
}

impl Eq for HashedCacheKeyRef<'_> {}

impl Hash for HashedCacheKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text_hash.hash(state);
        self.text_len.hash(state);
        self.font_size.hash(state);
        self.runs.hash(state);
        self.wrap_width.hash(state);
        self.force_width.hash(state);
        self.snap_tolerance.hash(state);
        self.snap_mode.hash(state);
    }
}

impl Eq for dyn AsCacheKeyRef + '_ {}

impl Hash for dyn AsCacheKeyRef + '_ {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state)
    }
}

impl AsCacheKeyRef for CacheKey {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        CacheKeyRef {
            text: &self.text,
            font_size: self.font_size,
            runs: self.runs.as_slice(),
            wrap_width: self.wrap_width,
            force_width: self.force_width,
            snap_tolerance: self.snap_tolerance,
            snap_mode: self.snap_mode.as_ref(),
        }
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_cache_key_ref().eq(&other.as_cache_key_ref())
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_cache_key_ref().hash(state);
    }
}

impl<'a> Borrow<dyn AsCacheKeyRef + 'a> for Arc<CacheKey> {
    fn borrow(&self) -> &(dyn AsCacheKeyRef + 'a) {
        self.as_ref() as &dyn AsCacheKeyRef
    }
}

impl AsCacheKeyRef for CacheKeyRef<'_> {
    fn as_cache_key_ref(&self) -> CacheKeyRef<'_> {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GlyphId, point};

    fn make_glyph(id: u32, index: usize, x: f32) -> ShapedGlyph {
        ShapedGlyph {
            id: GlyphId(id),
            position: point(px(x), px(0.)),
            index,
            is_emoji: false,
        }
    }

    fn single_run(glyphs: Vec<ShapedGlyph>) -> LineLayout {
        LineLayout {
            font_size: px(12.),
            width: px(0.),
            ascent: px(0.),
            descent: px(0.),
            runs: vec![ShapedRun {
                font_id: FontId(0),
                glyphs,
            }],
            len: 0,
        }
    }

    #[test]
    fn snap_glyph_index_legacy_tolerance() {
        // Three glyphs at natural positions slightly off the grid, tolerance = 1px.
        // With legacy 1px tolerance, small drifts are not snapped.
        let cell = px(7.);
        let mut layout = single_run(vec![
            make_glyph(1, 0, 0.0),
            make_glyph(2, 1, 7.2),
            make_glyph(3, 2, 14.4),
        ]);
        layout.len = 3;
        snap_glyph_positions(&mut layout, cell, px(1.), &SnapMode::GlyphIndex, 3);
        let run = &layout.runs[0];
        assert_eq!(run.glyphs[0].position.x, px(0.0));
        assert_eq!(run.glyphs[1].position.x, px(7.2));
        assert_eq!(run.glyphs[2].position.x, px(14.4));
        assert_eq!(layout.width, px(21.));
    }

    #[test]
    fn snap_glyph_index_strict_tolerance() {
        // With tolerance = 0, every glyph snaps to its GlyphIndex target.
        let cell = px(7.);
        let mut layout = single_run(vec![
            make_glyph(1, 0, 0.0),
            make_glyph(2, 1, 7.2246),
            make_glyph(3, 2, 14.4492),
        ]);
        layout.len = 3;
        snap_glyph_positions(&mut layout, cell, px(0.), &SnapMode::GlyphIndex, 3);
        let run = &layout.runs[0];
        assert_eq!(run.glyphs[0].position.x, px(0.));
        assert_eq!(run.glyphs[1].position.x, px(7.));
        assert_eq!(run.glyphs[2].position.x, px(14.));
        assert_eq!(layout.width, px(21.));
    }

    #[test]
    fn snap_byte_to_column_ligature() {
        // Simulated `==` ligature: 1 glyph spanning 2 bytes (2 columns),
        // followed by one glyph at byte index 2.
        // byte_to_column: [0, 1, 2, 3] — byte 0 → column 0, byte 2 → column 2.
        // force_width = 7px, text_len = 3.
        let column_width = px(7.);
        let mut layout = single_run(vec![
            make_glyph(1, 0, 0.0),   // ligature glyph at byte 0
            make_glyph(2, 2, 14.45), // next glyph at byte 2, natural advance ~14.45
        ]);
        layout.len = 3;
        let byte_to_column: Arc<[u32]> = Arc::from(vec![0, 1, 2, 3]);
        snap_glyph_positions(
            &mut layout,
            column_width,
            px(0.),
            &SnapMode::ByteToColumn { byte_to_column },
            3,
        );
        let run = &layout.runs[0];
        assert_eq!(run.glyphs[0].position.x, px(0.));
        // Next glyph snaps to byte_to_column[2] × column_width = 2 × 7 = 14, NOT 1 × 7 = 7.
        assert_eq!(run.glyphs[1].position.x, px(14.));
        // layout.width = byte_to_column[text_len] × column_width = 3 × 7 = 21.
        assert_eq!(layout.width, px(21.));
    }

    #[test]
    fn snap_byte_to_column_multi_byte_non_ascii() {
        // Text "éa": `é` = 2 bytes (width-1 column), `a` = 1 byte (width-1 column).
        // byte_to_column = [0, 0, 1, 2]: both bytes of `é` map to column 0, `a` byte to column 1.
        // text_len = 3, column_width = 7px.
        let column_width = px(7.);
        let mut layout = single_run(vec![
            make_glyph(1, 0, 0.0),  // `é` glyph at byte 0 → column 0
            make_glyph(2, 2, 7.22), // `a` glyph at byte 2 → column 1
        ]);
        layout.len = 3;
        let byte_to_column: Arc<[u32]> = Arc::from(vec![0, 0, 1, 2]);
        snap_glyph_positions(
            &mut layout,
            column_width,
            px(0.),
            &SnapMode::ByteToColumn { byte_to_column },
            3,
        );
        let run = &layout.runs[0];
        assert_eq!(run.glyphs[0].position.x, px(0.)); // column 0
        assert_eq!(run.glyphs[1].position.x, px(7.)); // column 1
        assert_eq!(layout.width, px(14.)); // byte_to_column[3] × 7 = 2 × 7
    }

    #[test]
    fn snap_byte_to_column_wide_multi_glyph_cluster() {
        // Uniform-grid usage: 3 glyphs for a 12-byte cluster (e.g. emoji ZWJ)
        // that spans 2 unit-width columns. force_width = base column width (7),
        // table = [0; 12] + sentinel 2. All 3 glyphs land at x = 0.
        // layout.width = 2 × 7 = 14.
        let column_width = px(7.);
        let mut layout = single_run(vec![
            make_glyph(1, 0, 0.0),
            make_glyph(2, 4, 20.0),
            make_glyph(3, 8, 40.0),
        ]);
        layout.len = 12;
        let mut table = vec![0u32; 12];
        table.push(2);
        let byte_to_column: Arc<[u32]> = Arc::from(table);
        snap_glyph_positions(
            &mut layout,
            column_width,
            px(0.),
            &SnapMode::ByteToColumn { byte_to_column },
            12,
        );
        let run = &layout.runs[0];
        assert_eq!(run.glyphs[0].position.x, px(0.));
        assert_eq!(run.glyphs[1].position.x, px(0.));
        assert_eq!(run.glyphs[2].position.x, px(0.));
        assert_eq!(layout.width, px(14.));
    }

    #[test]
    fn shape_line_options_normalized_collapses_when_force_width_none() {
        // Non-default snap fields with force_width=None must collapse to defaults
        // so cache hits aren't fragmented on irrelevant fields.
        let byte_to_column: Arc<[u32]> = Arc::from(vec![0, 1, 2]);
        let opts = ShapeLineOptions {
            force_width: None,
            snap_tolerance: px(0.),
            snap_mode: SnapMode::ByteToColumn { byte_to_column },
        };
        let normalized = opts.normalized();
        assert_eq!(normalized.force_width, None);
        assert_eq!(normalized.snap_tolerance, px(1.));
        assert!(matches!(normalized.snap_mode, SnapMode::GlyphIndex));

        // Force-width set: snap fields are preserved.
        let byte_to_column: Arc<[u32]> = Arc::from(vec![0, 1, 2]);
        let opts = ShapeLineOptions {
            force_width: Some(px(7.)),
            snap_tolerance: px(0.),
            snap_mode: SnapMode::ByteToColumn {
                byte_to_column: byte_to_column.clone(),
            },
        };
        let normalized = opts.normalized();
        assert_eq!(normalized.force_width, Some(px(7.)));
        assert_eq!(normalized.snap_tolerance, px(0.));
        match normalized.snap_mode {
            SnapMode::ByteToColumn {
                byte_to_column: got,
            } => {
                assert_eq!(&got[..], &byte_to_column[..]);
            }
            _ => panic!("snap_mode was not preserved"),
        }
    }
}
