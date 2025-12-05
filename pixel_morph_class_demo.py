import math
import random
import colorsys
import pygame

W, H = 1120, 700
BG = (12, 14, 22)
PANEL = (18, 20, 32)
TXT = (240, 240, 245)
MUTED = (190, 195, 210)
EDGE = (90, 95, 120)
OK = (120, 240, 170)
BAD = (255, 170, 130)

CELL = 120
GAP = 16
GRID = 2

SRC_ORG = (70, 210)
TGT_ORG = (500, 210)

BOTTOM_BAR_H = 62
TOP_MARGIN = 18

DARK_RED   = (140, 20, 35)
LIGHT_BLUE = (140, 180, 255)
DARK_BLUE  = (15, 35, 120)
LIGHT_RED  = (235, 120, 135)

TARGET = [
    DARK_RED,     # y0 (TL)
    LIGHT_BLUE,   # y1 (TR)
    DARK_BLUE,    # y2 (BL)
    LIGHT_RED,    # y3 (BR)
]

SOURCE_BASE = [DARK_RED, LIGHT_BLUE, DARK_BLUE, DARK_RED]


def rgb_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def hue_deg(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return (h * 360.0) % 360.0

def color_family(rgb):
    """
    Demo-friendly hue buckets so "reds prefer reds" and "blues prefer blues"
    (matches your explanation).
    """
    h = hue_deg(rgb)
    if h <= 30 or h >= 330:
        return "red"
    if 180 <= h <= 280:
        return "blue"
    return "other"

def cell_rect(origin, idx):
    r = idx // GRID
    c = idx % GRID
    x0, y0 = origin
    x = x0 + c * (CELL + GAP)
    y = y0 + r * (CELL + GAP)
    return pygame.Rect(x, y, CELL, CELL)

def wrap_lines(font, text, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def importance_by_focus(focus_px):
    fx, fy = focus_px
    scores = []
    for y in range(4):
        r = cell_rect(TGT_ORG, y)
        cx, cy = r.center
        d = math.hypot(cx - fx, cy - fy)
        scores.append(-d)  # closer => higher importance
    return scores

def build_costs(src_by_id, tgt):
    return [[rgb_dist(tgt[y], src_by_id[x]) for x in range(4)] for y in range(4)]

def preference_order_for_y(src_by_id, tgt, y):
    """
    FIXED: preference ordering is now "same hue-family first, then RGB distance"
    so it matches your intended story.
    """
    fy = color_family(tgt[y])
    xs = list(range(4))
    xs.sort(key=lambda x: (
        0 if color_family(src_by_id[x]) == fy else 1,
        rgb_dist(tgt[y], src_by_id[x])
    ))
    return xs

def simulate_sequential(src_by_id, tgt, imp_scores):
    order = sorted(range(4), key=lambda y: imp_scores[y], reverse=True)
    used = set()
    match = {}
    C = build_costs(src_by_id, tgt)

    for y in order:
        for x in preference_order_for_y(src_by_id, tgt, y):
            if x not in used:
                match[y] = x
                used.add(x)
                break
    return match, 0, C

def group_targets_naive():
    return [[0, 2], [1, 3]]

def group_targets_hue(tgt):
    reds, blues, other = [], [], []
    for y in range(4):
        fam = color_family(tgt[y])
        if fam == "red":
            reds.append(y)
        elif fam == "blue":
            blues.append(y)
        else:
            other.append(y)
    for y in other:
        (reds if len(reds) <= len(blues) else blues).append(y)
    return [reds, blues]

def simulate_parallel_local_then_merge(src_by_id, tgt, imp_scores, groups, do_repair):
    C = build_costs(src_by_id, tgt)

    stage1 = {}
    for g in groups:
        used_local = set()
        ys = sorted(g, key=lambda y: imp_scores[y], reverse=True)
        for y in ys:
            for x in preference_order_for_y(src_by_id, tgt, y):
                if x not in used_local:
                    stage1[y] = x
                    used_local.add(x)
                    break

    xs = list(stage1.values())
    dups = len(xs) - len(set(xs))

    if not do_repair:
        # drop duplicates by importance -> shows holes
        final = {}
        used_global = set()
        order = sorted(stage1.keys(), key=lambda y: imp_scores[y], reverse=True)
        for y in order:
            x = stage1[y]
            if x in used_global:
                continue
            final[y] = x
            used_global.add(x)
        return final, dups, C

    # merge + repair
    final = {}
    used_global = set()
    order_all = sorted(range(4), key=lambda y: imp_scores[y], reverse=True)

    for y in order_all:
        if y in stage1:
            x = stage1[y]
            if x not in used_global:
                final[y] = x
                used_global.add(x)

    for y in order_all:
        if y in final:
            continue
        for x in preference_order_for_y(src_by_id, tgt, y):
            if x not in used_global:
                final[y] = x
                used_global.add(x)
                break

    return final, dups, C

# ---------------- Tiles ----------------

class Tile:
    def __init__(self, tid, color, home_pos):
        self.tid = tid
        self.color = color
        self.home = home_pos
        self.rect = pygame.Rect(home_pos[0], home_pos[1], CELL, CELL)
        self.drag = False
        self.off = (0, 0)
        self.assigned_y = None

    def start_drag(self, pos):
        self.drag = True
        self.off = (self.rect.x - pos[0], self.rect.y - pos[1])

    def move_drag(self, pos):
        if self.drag:
            self.rect.x = pos[0] + self.off[0]
            self.rect.y = pos[1] + self.off[1]

    def stop_drag(self):
        self.drag = False

    def snap_to(self, pos):
        self.rect.topleft = pos

    def reset_home(self):
        self.assigned_y = None
        self.snap_to(self.home)

# ---------------- Main UI ----------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("2×2 Pixel Morph Demo: Sequential → Naive Parallel → Hue Grouped")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("menlo", 16)
    font_b = pygame.font.SysFont("menlo", 18, bold=True)
    font_h = pygame.font.SysFont("menlo", 22, bold=True)

    tgt_all = pygame.Rect(TGT_ORG[0], TGT_ORG[1], 2*CELL + GAP, 2*CELL + GAP)
    focus = tgt_all.center
    selected_y = None

    src_colors = SOURCE_BASE[:]
    random.shuffle(src_colors)

    tiles = []
    for i in range(4):
        r = cell_rect(SRC_ORG, i)
        tiles.append(Tile(i, src_colors[i], r.topleft))

    y_to_x = {}

    mode = "seq"   # "seq", "naive", "hue"
    conflict_mode = False

    msg = "Click a target cell (y0–y3) to show its preference ranking on the x tiles (badges 1–4)."

    def make_btn(x, y, w, h, label):
        return {"rect": pygame.Rect(x, y, w, h), "label": label}

    buttons = {
        "seq":   make_btn(840, 190, 240, 44, "Mode: Sequential"),
        "naive": make_btn(840, 242, 240, 44, "Mode: Parallel (Naive)"),
        "hue":   make_btn(840, 294, 240, 44, "Mode: Parallel (Hue)"),
        "sim":   make_btn(840, 362, 240, 44, "Simulate Stage"),
        "reset": make_btn(840, 414, 240, 44, "Reset Tiles"),
        "conf":  make_btn(840, 484, 240, 44, "Conflict Mode: OFF"),
        "resh":  make_btn(840, 536, 240, 44, "Reshuffle Source"),
    }

    def src_by_id():
        arr = [None]*4
        for t in tiles:
            arr[t.tid] = t.color
        return arr

    def tile_by_id(tid):
        for t in tiles:
            if t.tid == tid:
                return t
        return None

    def clear_assignments():
        nonlocal y_to_x
        y_to_x = {}
        for t in tiles:
            t.reset_home()

    def recompute_y_to_x_from_tiles():
        nonlocal y_to_x
        y_to_x = {}
        for t in tiles:
            if t.assigned_y is not None:
                y_to_x[t.assigned_y] = t.tid

    def handle_drop(tile):
        nonlocal y_to_x
        placed = False
        for y in range(4):
            r = cell_rect(TGT_ORG, y)
            if r.collidepoint(tile.rect.center):
                if y in y_to_x:
                    old_tid = y_to_x[y]
                    if old_tid != tile.tid:
                        tile_by_id(old_tid).reset_home()
                tile.assigned_y = y
                tile.snap_to(r.topleft)
                y_to_x[y] = tile.tid
                placed = True
                break
        if not placed:
            tile.reset_home()
        recompute_y_to_x_from_tiles()

    def update_conflict_sources():
        nonlocal src_colors
        if not conflict_mode:
            src_colors = SOURCE_BASE[:]
        else:
            # make red scarce -> naive parallel clashes easier to show
            src_colors = [DARK_RED, LIGHT_BLUE, DARK_BLUE, LIGHT_BLUE]
        random.shuffle(src_colors)
        for i in range(4):
            tile_by_id(i).color = src_colors[i]
        clear_assignments()

    def draw_button(b, active=False):
        r = b["rect"]
        fill = (34, 38, 60) if not active else (44, 56, 88)
        pygame.draw.rect(screen, fill, r, border_radius=10)
        pygame.draw.rect(screen, EDGE, r, width=2, border_radius=10)
        screen.blit(font_b.render(b["label"], True, TXT), (r.x + 12, r.y + 10))

    def draw_group_overlay(mode_now):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)

        if mode_now == "naive":
            left = pygame.Rect(TGT_ORG[0]-8, TGT_ORG[1]-8, CELL+16, 2*CELL + GAP + 16)
            right = pygame.Rect(TGT_ORG[0] + CELL + GAP - 8, TGT_ORG[1]-8, CELL+16, 2*CELL + GAP + 16)
            pygame.draw.rect(overlay, (120, 200, 255, 200), left, width=6, border_radius=18)
            pygame.draw.rect(overlay, (255, 160, 120, 200), right, width=6, border_radius=18)

        if mode_now == "hue":
            def ring_cell(y, rgba, pad=10, w=6):
                r = cell_rect(TGT_ORG, y).inflate(pad*2, pad*2)
                pygame.draw.rect(overlay, rgba, r, width=w, border_radius=18)
                return r.center

            def link(a, b, rgba, w=12):
                pygame.draw.line(overlay, rgba, a, b, width=w)
                pygame.draw.circle(overlay, rgba, a, w//2 + 2)
                pygame.draw.circle(overlay, rgba, b, w//2 + 2)

            red_cells = [y for y in range(4) if color_family(TARGET[y]) == "red"]
            blue_cells = [y for y in range(4) if color_family(TARGET[y]) == "blue"]

            if len(red_cells) >= 2:
                a = ring_cell(red_cells[0], (255, 160, 120, 210))
                b = ring_cell(red_cells[1], (255, 160, 120, 210))
                link(a, b, (255, 160, 120, 160))

            if len(blue_cells) >= 2:
                a = ring_cell(blue_cells[0], (120, 200, 255, 210))
                b = ring_cell(blue_cells[1], (120, 200, 255, 210))
                link(a, b, (120, 200, 255, 160))

        screen.blit(overlay, (0, 0))

    def preference_rank_map_for_selected_y():
        if selected_y is None:
            return None, None
        src = src_by_id()
        C = build_costs(src, TARGET)
        order = preference_order_for_y(src, TARGET, selected_y)
        rank = {x: i+1 for i, x in enumerate(order)}
        return rank, C

    def draw_rank_badge(tile_rect, rank_num):
        cx = tile_rect.right - 18
        cy = tile_rect.top + 18
        pygame.draw.circle(screen, (250, 250, 255), (cx, cy), 14)
        pygame.draw.circle(screen, (20, 20, 30), (cx, cy), 14, 2)
        lab = font_b.render(str(rank_num), True, (20, 20, 30))
        screen.blit(lab, (cx - lab.get_width()//2, cy - lab.get_height()//2))

    def draw():
        screen.fill(BG)

        # Header
        screen.blit(font_h.render("2×2 Pixel Morph Demo", True, TXT), (70, TOP_MARGIN))
        sub1 = "Sequential → Naive parallel (split) → Hue-grouped parallel (reds+blues)."
        sub2 = "Click a target cell to select y; x tiles show preference rank (1=best)."
        screen.blit(font.render(sub1, True, MUTED), (70, TOP_MARGIN + 30))
        screen.blit(font.render(sub2, True, MUTED), (70, TOP_MARGIN + 52))

        # Labels
        screen.blit(font_b.render("Image A (Source tiles)", True, TXT), (SRC_ORG[0], 165))
        screen.blit(font_b.render("Image B (Target colors)", True, TXT), (TGT_ORG[0], 165))

        # Target cells + importance ranks
        imp = importance_by_focus(focus)
        imp_order = sorted(range(4), key=lambda yy: imp[yy], reverse=True)

        for yidx in range(4):
            rt = cell_rect(TGT_ORG, yidx)
            ghost = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            ghost.fill((*TARGET[yidx], 165))
            screen.blit(ghost, rt.topleft)
            pygame.draw.rect(screen, (235, 235, 245), rt, width=2, border_radius=12)

            # importance rank
            rank_num = imp_order.index(yidx) + 1
            num = font_b.render(str(rank_num), True, (20, 20, 28))
            screen.blit(num, (rt.x + 10, rt.y + 8))

            # y label
            yl = font.render(f"y{yidx}", True, TXT)
            screen.blit(yl, (rt.x + 10, rt.y + CELL - 24))

            # selected highlight
            if selected_y == yidx:
                pygame.draw.rect(screen, (255, 255, 255), rt, width=4, border_radius=12)

        # focus marker
        pygame.draw.circle(screen, (255, 255, 255), focus, 5)
        pygame.draw.circle(screen, (0, 0, 0), focus, 5, 2)

        # grouping overlay only for parallel modes
        if mode in ("naive", "hue"):
            draw_group_overlay(mode)

        # Source outlines
        for i in range(4):
            rs = cell_rect(SRC_ORG, i)
            pygame.draw.rect(screen, (65, 70, 90), rs, width=2, border_radius=12)

        # Preference badges for selected y
        rank_map, C_sel = preference_rank_map_for_selected_y()

        # Draw tiles
        for t in tiles:
            pygame.draw.rect(screen, t.color, t.rect, border_radius=12)
            pygame.draw.rect(screen, (245, 245, 250), t.rect, width=3, border_radius=12)

            # x label
            tl_col = (15, 15, 24) if sum(t.color) > 520 else (245, 245, 250)
            screen.blit(font_b.render(f"x{t.tid}", True, tl_col), (t.rect.x + 10, t.rect.y + 8))

            # rank badge
            if rank_map is not None and t.tid in rank_map:
                draw_rank_badge(t.rect, rank_map[t.tid])

        # Right panel
        pygame.draw.rect(screen, PANEL, pygame.Rect(820, 150, 280, 490), border_radius=16)
        screen.blit(font_b.render("Controls", True, TXT), (840, 162))

        draw_button(buttons["seq"],   active=(mode == "seq"))
        draw_button(buttons["naive"], active=(mode == "naive"))
        draw_button(buttons["hue"],   active=(mode == "hue"))
        draw_button(buttons["sim"])
        draw_button(buttons["reset"])
        draw_button(buttons["conf"])
        draw_button(buttons["resh"])

        # Status
        Call = build_costs(src_by_id(), TARGET)
        total = sum(Call[y][x] for y, x in y_to_x.items())
        complete = (len(y_to_x) == 4)
        bij = complete and (len(set(y_to_x.values())) == 4)

        sy = 30
        screen.blit(font_b.render("Status", True, TXT), (840, sy))
        screen.blit(font.render(f"Placed: {len(y_to_x)}/4", True, MUTED), (840, sy + 24))
        screen.blit(font.render(f"Bijection complete: {'YES' if bij else 'no'}", True, OK if bij else BAD), (840, sy + 44))
        screen.blit(font.render(f"Total distance: {total:.1f}", True, MUTED), (840, sy + 64))

        # Selected y info
        iy = 560
        if selected_y is None:
            screen.blit(font.render("Selected y: (none)", True, MUTED), (840, iy))
        else:
            screen.blit(font.render(f"Selected y: y{selected_y}", True, MUTED), (840, iy))
            if C_sel is not None:
                best_x = preference_order_for_y(src_by_id(), TARGET, selected_y)[0]
                screen.blit(font.render(f"Rank #1: x{best_x}", True, MUTED), (840, iy + 20))

        # Bottom message bar
        box = pygame.Rect(70, H - BOTTOM_BAR_H, W - 90, BOTTOM_BAR_H - 12)
        pygame.draw.rect(screen, (28, 30, 46), box, border_radius=12)
        pygame.draw.rect(screen, EDGE, box, width=2, border_radius=12)
        lines = wrap_lines(font, msg, box.w - 20)
        yy = box.y + 10
        for line in lines[:2]:
            screen.blit(font.render(line, True, TXT), (box.x + 12, yy))
            yy += 18

        pygame.display.flip()

    dragging = None
    running = True
    while running:
        clock.tick(60)

        buttons["conf"]["label"] = "Conflict Mode: ON" if conflict_mode else "Conflict Mode: OFF"

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                pos = e.pos

                # Click target cell: select y + move focus
                if tgt_all.collidepoint(pos):
                    clicked = None
                    for y in range(4):
                        if cell_rect(TGT_ORG, y).collidepoint(pos):
                            clicked = y
                            break
                    if clicked is not None:
                        selected_y = clicked
                    focus = pos

                # Buttons
                if buttons["seq"]["rect"].collidepoint(pos):
                    mode = "seq"
                    msg = "Mode: Sequential (no grouping overlay)."
                elif buttons["naive"]["rect"].collidepoint(pos):
                    mode = "naive"
                    msg = "Mode: Parallel naive (split left vs right)."
                elif buttons["hue"]["rect"].collidepoint(pos):
                    mode = "hue"
                    msg = "Mode: Parallel hue-grouped (reds together, blues together)."
                elif buttons["reset"]["rect"].collidepoint(pos):
                    clear_assignments()
                    msg = "Reset tiles."
                elif buttons["resh"]["rect"].collidepoint(pos):
                    cols = [tile_by_id(i).color for i in range(4)]
                    random.shuffle(cols)
                    for i in range(4):
                        tile_by_id(i).color = cols[i]
                    clear_assignments()
                    msg = "Reshuffled source tiles."
                elif buttons["conf"]["rect"].collidepoint(pos):
                    conflict_mode = not conflict_mode
                    update_conflict_sources()
                    msg = "Conflict mode toggled (makes naive clashes easier to show)."
                elif buttons["sim"]["rect"].collidepoint(pos):
                    imp = importance_by_focus(focus)
                    src = src_by_id()

                    if mode == "seq":
                        final, _, _ = simulate_sequential(src, TARGET, imp)
                        clear_assignments()
                        for y, x in final.items():
                            t = tile_by_id(x)
                            t.assigned_y = y
                            t.snap_to(cell_rect(TGT_ORG, y).topleft)
                            y_to_x[y] = x
                        msg = "Sequential simulated: global used-set (no clashes)."

                    elif mode == "naive":
                        groups = group_targets_naive()
                        final, dups, _ = simulate_parallel_local_then_merge(src, TARGET, imp, groups, do_repair=False)
                        clear_assignments()
                        for y, x in final.items():
                            t = tile_by_id(x)
                            t.assigned_y = y
                            t.snap_to(cell_rect(TGT_ORG, y).topleft)
                            y_to_x[y] = x
                        msg = "Naive parallel simulated: CLASH ⇒ holes visible." if dups > 0 else "Naive parallel simulated: no clash this time."

                    else:
                        groups = group_targets_hue(TARGET)
                        final, _, _ = simulate_parallel_local_then_merge(src, TARGET, imp, groups, do_repair=True)
                        clear_assignments()
                        for y, x in final.items():
                            t = tile_by_id(x)
                            t.assigned_y = y
                            t.snap_to(cell_rect(TGT_ORG, y).topleft)
                            y_to_x[y] = x
                        msg = "Hue parallel simulated: grouped local work + merge/repair."

                else:
                    # Drag tile
                    for t in reversed(tiles):
                        if t.rect.collidepoint(pos):
                            dragging = t
                            if t.assigned_y is not None:
                                if t.assigned_y in y_to_x and y_to_x[t.assigned_y] == t.tid:
                                    del y_to_x[t.assigned_y]
                                t.assigned_y = None
                            t.start_drag(pos)
                            tiles.remove(t)
                            tiles.append(t)
                            break

            if e.type == pygame.MOUSEMOTION and dragging is not None:
                dragging.move_drag(e.pos)

            if e.type == pygame.MOUSEBUTTONUP and e.button == 1 and dragging is not None:
                dragging.stop_drag()
                handle_drop(dragging)
                dragging = None

        draw()

    pygame.quit()

if __name__ == "__main__":
    main()
