#!/usr/bin/env python3
"""
GAMEBOYADVANCEEMU4K — GBA shell with optional **Cython** render helpers.

PR / theme
------------
  * Blue-hue animated background (cool cyan → deep blue).
  * Buttons: black fill, thin blue outline; label text blue on black.
  * Body / status text: blue tones on the blue field for readability.

Engine scope (honest)
---------------------
  This is a **host + header + placeholder PPU** surface (240×160). It does
  not implement a full ARM7TDMI + GBA MMU + real PPU/DMA/Sound. Commercial
  games need mGBA, VBA-M, or a libretro core. The Cython module
  (`gba_render_fast`) accelerates RGB555→RGB888 and nearest-neighbor upscale
  when you plug in a real framebuffer producer later.

Optional Cython
---------------
    pip install cython
    python setup_gba.py build_ext --inplace

Requirements
------------
    pip install pygame numpy
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path
from typing import Optional

try:
    import pygame
except ImportError:
    print("pip install pygame", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    from gba_render_fast import rgb555_rows_to_rgb888 as _cy555  # type: ignore
    from gba_render_fast import scale_nn_240x160 as _cy_scale  # type: ignore

    _HAS_CY = True
except ImportError:
    _HAS_CY = False

try:
    import tkinter as tk
    from tkinter import filedialog

    _HAS_TK = True
except ImportError:
    _HAS_TK = False

# GBA LCD
GBA_W, GBA_H = 240, 160

# --- Theme: blue hue field, black buttons, blue text (# pr polish) ---
def _blue_field_rgb(tick_ms: float, u: float, v: float) -> tuple[int, int, int]:
    """u,v in 0..1 screen space. Returns soft blue-cyan background."""
    phase = (tick_ms * 0.012) + u * 4.2 + v * 3.1
    h = (200 + 55 * math.sin(phase)) % 360
    s = 0.55 + 0.12 * math.sin(phase * 0.7 + u * 6)
    val = 0.12 + 0.08 * math.sin(phase * 0.3 + v * 5)
    # HSV to RGB (mini)
    h60 = (h % 360) / 60
    i = int(h60)
    f = h60 - i
    p = val * (1 - s)
    q = val * (1 - s * f)
    t = val * (1 - s * (1 - f))
    if i == 0:
        r, g, b = val, t, p
    elif i == 1:
        r, g, b = q, val, p
    elif i == 2:
        r, g, b = p, val, t
    elif i == 3:
        r, g, b = p, q, val
    elif i == 4:
        r, g, b = t, p, val
    else:
        r, g, b = val, p, q
    return int(r * 255), int(g * 255), int(b * 255)


TEXT_BLUE = (120, 190, 255)
TEXT_BLUE_HI = (180, 220, 255)
TEXT_BLUE_DIM = (70, 130, 210)
BTN_BLACK = (8, 8, 12)
BTN_BORDER = (60, 140, 220)
ACCENT_LINE = (90, 170, 255)


def parse_gba_header(rom: bytes) -> dict:
    """Parse fixed GBA cartridge header (not full validation)."""
    if len(rom) < 0xC0:
        return {}
    entry = struct.unpack_from("<I", rom, 0)[0]
    title = rom[0xA0 : 0xAC].decode("ascii", errors="replace").rstrip("\x00 ")
    game_code = rom[0xAC : 0xB0].decode("ascii", errors="replace")
    maker = rom[0xB0 : 0xB2].hex()
    ver = rom[0xBC]
    chk = rom[0xBD]
    return {
        "entry": entry,
        "title": title or "(blank)",
        "game_code": game_code,
        "maker": maker,
        "version": ver,
        "hdr_checksum": chk,
        "size": len(rom),
    }


def _rgb555_to_rgb888_py(rom: bytes, tick_ms: int) -> bytes:
    """Placeholder 240×160 RGB555-ish demo pattern from ROM hash (Python path)."""
    if _HAS_NP:
        seed = sum(rom[:512]) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        row16 = rng.integers(0, 0x8000, size=(GBA_H, GBA_W), dtype=np.uint16)
        row16 ^= (tick_ms // 8) & 0xFFFF
        r = ((row16 >> 0) & 0x1F).astype(np.uint8) << 3
        g = ((row16 >> 5) & 0x1F).astype(np.uint8) << 3
        b = ((row16 >> 10) & 0x1F).astype(np.uint8) << 3
        return np.dstack([r, g, b]).tobytes()
    out = bytearray(GBA_W * GBA_H * 3)
    s = sum(rom[:256]) if rom else 1
    for y in range(GBA_H):
        for x in range(GBA_W):
            v = ((x * 13 + y * 7 + s + tick_ms // 4) ^ (x << 8)) & 0x7FFF
            r = ((v >> 0) & 0x1F) << 3
            g = ((v >> 5) & 0x1F) << 3
            b = ((v >> 10) & 0x1F) << 3
            o = (y * GBA_W + x) * 3
            out[o + 0] = r
            out[o + 1] = g
            out[o + 2] = b
    return bytes(out)


def _rgb555_pitch_buffer(tick_ms: int, rom: bytes) -> tuple[bytes, int]:
    """Build tight RGB555 240×160 then convert via Cython or numpy."""
    if _HAS_NP:
        seed = sum(rom[:512]) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        row16 = rng.integers(0, 0x8000, size=(GBA_H, GBA_W), dtype="<u2")
        row16 ^= np.uint16((tick_ms // 8) & 0xFFFF)
        packed = row16.tobytes()
        pitch = GBA_W * 2
        if _HAS_CY:
            return _cy555(packed, GBA_W, GBA_H, pitch), GBA_W * 3
        flat = row16.reshape(-1)
        r = ((flat >> 0) & 0x1F).astype(np.uint8) << 3
        g = ((flat >> 5) & 0x1F).astype(np.uint8) << 3
        b = ((flat >> 10) & 0x1F).astype(np.uint8) << 3
        return np.column_stack([r, g, b]).tobytes(), GBA_W * 3
    rgb = _rgb555_to_rgb888_py(rom, tick_ms)
    return rgb, GBA_W * 3


def _scale_to(rgb: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes:
    if _HAS_CY:
        return _cy_scale(rgb, sw, sh, dw, dh)
    if _HAS_NP:
        src = np.frombuffer(rgb, dtype=np.uint8).reshape(sh, sw, 3)
        y_idx = (np.arange(dh) * (sh - 1) // max(1, dh - 1)).clip(0, sh - 1)
        x_idx = (np.arange(dw) * (sw - 1) // max(1, dw - 1)).clip(0, sw - 1)
        return src[y_idx[:, None], x_idx].reshape(-1).tobytes()
    out = bytearray(dw * dh * 3)
    for y in range(dh):
        sy = (y * (sh - 1)) // max(1, dh - 1) if dh > 1 else 0
        for x in range(dw):
            sx = (x * (sw - 1)) // max(1, dw - 1) if dw > 1 else 0
            si = (sy * sw + sx) * 3
            di = (y * dw + x) * 3
            out[di : di + 3] = rgb[si : si + 3]
    return bytes(out)


def draw_round_rect(surf, rect, fill, border=None, bw=1):
    pygame.draw.rect(surf, fill, rect, border_radius=8)
    if border:
        pygame.draw.rect(surf, border, rect, bw, border_radius=8)


def draw_btn(surf, font, label: str, rect: pygame.Rect, mouse, enabled=True):
    hov = enabled and rect.collidepoint(mouse)
    fill = (18, 22, 28) if hov else BTN_BLACK
    draw_round_rect(surf, rect, fill, BTN_BORDER if enabled else (40, 50, 60), 1)
    col = TEXT_BLUE_HI if hov else (TEXT_BLUE if enabled else TEXT_BLUE_DIM)
    t = font.render(label, True, col)
    surf.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("GAMEBOYADVANCEEMU4K — Cython GBA host (demo PPU)")

    w, h = 920, 620
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    try:
        font_title = pygame.font.SysFont("consolas", 20, bold=True)
        font_body = pygame.font.SysFont("consolas", 14)
        font_small = pygame.font.SysFont("consolas", 12)
    except Exception:
        font_title = font_body = font_small = pygame.font.Font(None, 18)

    rom: bytes = b""
    hdr: dict = {}
    root = None
    if _HAS_TK:
        root = tk.Tk()
        root.withdraw()

    margin = 16
    header_h = 44
    view = pygame.Rect(margin, header_h + margin, 720, 480)
    side = pygame.Rect(view.right + margin, view.y, w - view.right - 2 * margin, view.height)

    bar_y = h - 52
    b_load = pygame.Rect(margin, bar_y, 140, 40)
    b_run = pygame.Rect(margin + 152, bar_y, 100, 40)
    b_quit = pygame.Rect(margin + 264, bar_y, 100, 40)

    running = True
    auto_demo = True
    toast = ""
    toast_t = 0

    def toast_msg(m: str, t: int = 90):
        nonlocal toast, toast_t
        toast, toast_t = m, t

    while running:
        tick = pygame.time.get_ticks()
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if b_load.collidepoint(ev.pos):
                    if not _HAS_TK:
                        toast_msg("tkinter missing", 120)
                    else:
                        p = filedialog.askopenfilename(
                            parent=root,
                            title="Open GBA ROM",
                            filetypes=[("GBA ROM", "*.gba *.GBA"), ("All", "*.*")],
                        )
                        if p:
                            try:
                                rom = Path(p).read_bytes()
                            except OSError as e:
                                toast_msg(str(e), 140)
                            else:
                                hdr = parse_gba_header(rom)
                                toast_msg(f"loaded {hdr.get('title', '?')[:24]}", 120)
                                auto_demo = False
                elif b_run.collidepoint(ev.pos):
                    auto_demo = not auto_demo
                    toast_msg("demo PPU " + ("on" if auto_demo else "paused"), 80)
                elif b_quit.collidepoint(ev.pos):
                    running = False

        # animated blue field
        for yy in range(0, h, 6):
            for xx in range(0, w, 6):
                c = _blue_field_rgb(float(tick), xx / max(1, w - 1), yy / max(1, h - 1))
                pygame.draw.rect(screen, c, (xx, yy, 6, 6))

        # header strip
        hdr_rect = pygame.Rect(0, 0, w, header_h)
        draw_round_rect(screen, hdr_rect, BTN_BLACK, ACCENT_LINE, 2)
        sub = f"{'Cython ON' if _HAS_CY else 'Cython OFF (pip install cython + build)'}  ·  numpy={'yes' if _HAS_NP else 'no'}"
        t1 = font_title.render("GAMEBOYADVANCEEMU4K", True, TEXT_BLUE_HI)
        t2 = font_title.render(sub, True, TEXT_BLUE_DIM)
        screen.blit(t1, (margin, 10))
        screen.blit(t2, (margin + t1.get_width() + 12, 14))

        # viewport frame (LCD bezel)
        draw_round_rect(screen, view, (10, 14, 22), ACCENT_LINE, 2)
        inner = view.inflate(-24, -24)
        pygame.draw.rect(screen, (0, 0, 0), inner)

        if rom:
            frame_tick = tick if auto_demo else 0
            rgb888, _ = _rgb555_pitch_buffer(frame_tick, rom)
            iw, ih = inner.w, inner.h
            scaled = _scale_to(rgb888, GBA_W, GBA_H, iw, ih)
            try:
                surf = pygame.image.frombuffer(scaled, (iw, ih), "RGB")
                screen.blit(surf, inner.topleft)
            except Exception as e:
                err = font_body.render(f"blit: {e}", True, TEXT_BLUE)
                screen.blit(err, (inner.x + 8, inner.y + 8))
        else:
            hint = font_body.render("Load a .gba ROM — demo uses header + placeholder PPU", True, TEXT_BLUE)
            screen.blit(hint, (inner.x + 12, inner.y + 12))

        # side panel
        draw_round_rect(screen, side, (12, 16, 28), ACCENT_LINE, 1)
        sy = side.y + 10
        screen.blit(font_title.render("Cart header", True, TEXT_BLUE_HI), (side.x + 10, sy))
        sy += 26
        if hdr:
            for lab, key in [
                ("Title", "title"),
                ("Code", "game_code"),
                ("Maker", "maker"),
                ("Entry", "entry"),
                ("Ver", "version"),
                ("ROM", "size"),
            ]:
                screen.blit(font_small.render(f"{lab:6}", True, TEXT_BLUE_DIM), (side.x + 10, sy))
                val = str(hdr.get(key, ""))[:28]
                screen.blit(font_small.render(val, True, TEXT_BLUE), (side.x + 78, sy))
                sy += 16
        else:
            screen.blit(font_small.render("(no ROM)", True, TEXT_BLUE_DIM), (side.x + 10, sy))

        draw_btn(screen, font_body, "Load ROM…", b_load, mouse)
        draw_btn(screen, font_body, "Demo " + ("ON" if auto_demo else "OFF"), b_run, mouse, enabled=bool(rom))
        draw_btn(screen, font_body, "Exit", b_quit, mouse)

        if toast_t > 0:
            toast_t -= 1
            s = font_small.render(toast, True, TEXT_BLUE_HI)
            pygame.draw.rect(screen, BTN_BLACK, (10, h - 36, s.get_width() + 16, 28), border_radius=6)
            pygame.draw.rect(screen, BTN_BORDER, (10, h - 36, s.get_width() + 16, 28), 1, border_radius=6)
            screen.blit(s, (18, h - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    if root:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
