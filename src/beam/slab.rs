// src/beam/slab.rs
// =================
// Identical to the PyTorch version — the Rust slab allocator is
// completely framework-agnostic (pure C FFI, no torch/jax dependency).
//
// Build:  cargo build --release
// Output: target/release/libbeam_slab.so  (Linux)
//         target/release/libbeam_slab.dylib (macOS)

use std::sync::Mutex;

const HDR_FIXED: usize = 4;
const TAIL_BYTES: usize = 8 + 4 + 4;

fn slot_bytes(max_cdr_len: u32) -> usize {
    HDR_FIXED + (max_cdr_len as usize) * 4 + TAIL_BYTES
}

pub struct SlabArena {
    data:        Vec<u8>,
    free_list:   Mutex<Vec<usize>>,
    n_slots:     usize,
    max_cdr_len: u32,
    slot_bytes:  usize,
}

impl SlabArena {
    pub fn new(n_slots: u32, max_cdr_len: u32) -> Box<Self> {
        let sb    = slot_bytes(max_cdr_len);
        let total = (n_slots as usize) * sb;
        let data  = vec![0u8; total];
        let free  = (0..n_slots as usize).rev().collect::<Vec<_>>();
        Box::new(SlabArena {
            data,
            free_list: Mutex::new(free),
            n_slots:   n_slots as usize,
            max_cdr_len,
            slot_bytes: sb,
        })
    }

    pub fn alloc(&self) -> i64 {
        let mut fl = self.free_list.lock().unwrap();
        match fl.pop() {
            Some(idx) => {
                let off = idx * self.slot_bytes;
                let end = off + self.slot_bytes;
                self.data_slice_mut(off, end).fill(0);
                idx as i64
            }
            None => -1,
        }
    }

    pub fn free(&self, handle: i64) {
        if handle < 0 || handle as usize >= self.n_slots { return; }
        let mut fl = self.free_list.lock().unwrap();
        fl.push(handle as usize);
    }

    pub fn write_seq(&self, handle: i64, seq: &[i32]) {
        let base = self.slot_base(handle);
        let n    = seq.len().min(self.max_cdr_len as usize);
        self.write_u32(base, n as u32);
        for (i, &tok) in seq[..n].iter().enumerate() {
            self.write_i32(base + HDR_FIXED + i * 4, tok);
        }
    }

    pub fn read_seq(&self, handle: i64, out: &mut Vec<i32>) {
        let base = self.slot_base(handle);
        let n    = self.read_u32(base) as usize;
        out.clear();
        for i in 0..n { out.push(self.read_i32(base + HDR_FIXED + i * 4)); }
    }

    pub fn write_parent(&self, handle: i64, parent: i64) {
        self.write_i64(self.tail_offset(handle), parent);
    }
    pub fn read_parent(&self, handle: i64) -> i64 {
        self.read_i64(self.tail_offset(handle))
    }
    pub fn write_logprob(&self, handle: i64, lp: f32) {
        self.write_f32(self.tail_offset(handle) + 8, lp);
    }
    pub fn read_logprob(&self, handle: i64) -> f32 {
        self.read_f32(self.tail_offset(handle) + 8)
    }
    pub fn stats(&self) -> (u32, u32) {
        let fl     = self.free_list.lock().unwrap();
        let n_free = fl.len() as u32;
        (self.n_slots as u32 - n_free, n_free)
    }

    fn slot_base(&self, h: i64) -> usize { h as usize * self.slot_bytes }
    fn tail_offset(&self, h: i64) -> usize {
        self.slot_base(h) + HDR_FIXED + self.max_cdr_len as usize * 4
    }
    fn data_slice_mut(&self, s: usize, e: usize) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_ptr() as *mut u8, self.data.len()
            )[s..e].as_mut()
        }
    }
    fn write_u32(&self, o: usize, v: u32) {
        self.data_slice_mut(o, o+4).copy_from_slice(&v.to_le_bytes());
    }
    fn read_u32(&self, o: usize) -> u32 {
        u32::from_le_bytes(self.data[o..o+4].try_into().unwrap())
    }
    fn write_i32(&self, o: usize, v: i32) {
        self.data_slice_mut(o, o+4).copy_from_slice(&v.to_le_bytes());
    }
    fn read_i32(&self, o: usize) -> i32 {
        i32::from_le_bytes(self.data[o..o+4].try_into().unwrap())
    }
    fn write_i64(&self, o: usize, v: i64) {
        self.data_slice_mut(o, o+8).copy_from_slice(&v.to_le_bytes());
    }
    fn read_i64(&self, o: usize) -> i64 {
        i64::from_le_bytes(self.data[o..o+8].try_into().unwrap())
    }
    fn write_f32(&self, o: usize, v: f32) {
        self.data_slice_mut(o, o+4).copy_from_slice(&v.to_le_bytes());
    }
    fn read_f32(&self, o: usize) -> f32 {
        f32::from_le_bytes(self.data[o..o+4].try_into().unwrap())
    }
}

#[no_mangle]
pub extern "C" fn slab_create(n_slots: u32, max_cdr_len: u32) -> *mut SlabArena {
    Box::into_raw(SlabArena::new(n_slots, max_cdr_len))
}
#[no_mangle]
pub extern "C" fn slab_alloc(arena: *mut SlabArena) -> i64 {
    unsafe { (*arena).alloc() }
}
#[no_mangle]
pub extern "C" fn slab_free(arena: *mut SlabArena, handle: i64) {
    unsafe { (*arena).free(handle) }
}
#[no_mangle]
pub extern "C" fn slab_write_seq(
    arena: *mut SlabArena, handle: i64, seq: *const i32, length: u32,
) {
    let slice = unsafe { std::slice::from_raw_parts(seq, length as usize) };
    unsafe { (*arena).write_seq(handle, slice) };
}
#[no_mangle]
pub extern "C" fn slab_read_seq(
    arena: *mut SlabArena, handle: i64, out_ptr: *mut i32, out_len: *mut u32,
) {
    let mut buf = Vec::new();
    unsafe { (*arena).read_seq(handle, &mut buf) };
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), out_ptr, buf.len());
        *out_len = buf.len() as u32;
    }
}
#[no_mangle]
pub extern "C" fn slab_write_logprob(arena: *mut SlabArena, handle: i64, lp: f32) {
    unsafe { (*arena).write_logprob(handle, lp) }
}
#[no_mangle]
pub extern "C" fn slab_read_logprob(arena: *mut SlabArena, handle: i64) -> f32 {
    unsafe { (*arena).read_logprob(handle) }
}
#[no_mangle]
pub extern "C" fn slab_write_parent(arena: *mut SlabArena, handle: i64, parent: i64) {
    unsafe { (*arena).write_parent(handle, parent) }
}
#[no_mangle]
pub extern "C" fn slab_read_parent(arena: *mut SlabArena, handle: i64) -> i64 {
    unsafe { (*arena).read_parent(handle) }
}
#[no_mangle]
pub extern "C" fn slab_stats(arena: *mut SlabArena, n_used: *mut u32, n_free: *mut u32) {
    let (u, f) = unsafe { (*arena).stats() };
    unsafe { *n_used = u; *n_free = f; }
}
#[no_mangle]
pub extern "C" fn slab_destroy(arena: *mut SlabArena) {
    if !arena.is_null() { unsafe { drop(Box::from_raw(arena)) } }
}
