//! # Free-List Allocator
//!
//! Building on the bump allocator, implement a Free-List Allocator that supports memory reclamation.
//!
//! ## How It Works
//!
//! A Free-List Allocator uses a linked list to track all freed memory blocks.
//! On allocation, it first searches the list for a suitable block (first-fit strategy);
//! if none is found, it falls back to allocating from the unused region.
//! On deallocation, the block is inserted at the head of the list.
//!
//! ```text
//! free_list -> [block A: 64B] -> [block B: 128B] -> [block C: 32B] -> null
//! ```
//!
//! Each free block stores a `FreeBlock` struct at its head (containing block size and next pointer).
//!
//! ## Task
//!
//! Implement `FreeListAllocator`'s `alloc` and `dealloc` methods:
//!
//! ### alloc
//! 1. Traverse the free_list, find the first block with `size >= layout.size()` and proper alignment (first-fit)
//! 2. If found, remove it from the list and return it
//! 3. If not found, allocate from the `bump` region (same as bump allocator)
//!
//! ### dealloc
//! 1. Write `FreeBlock` header info at the freed block
//! 2. Insert it at the head of free_list
//!
//! ## Key Concepts
//!
//! - Intrusive linked list
//! - `*mut T` read/write: `ptr.write(val)` / `ptr.read()`
//! - Memory alignment checks

#![cfg_attr(not(test), no_std)]

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::Ordering;

/// Free block header, stored at the beginning of each free memory block.
///
/// `next` is a *relative* link (byte offset from `heap_start`, 0 = end of list) rather
/// than an absolute pointer, so "empty list" is just zero and a node never carries a raw
/// address that could go stale. `bias` records how many leading bytes of the block were
/// skipped when it was handed out with a stricter alignment than its own start provides.
struct FreeBlock {
    size: usize,
    next: usize,
    bias: usize,
}

/// Smallest usable block: one header. Anything smaller cannot re-enter the list.
const MIN_BLOCK: usize = core::mem::size_of::<FreeBlock>();

/// Sentinel value for "no successor" in a relative link. It must not be 0 because
/// offset 0 is a real, allocatable address (the first byte of the heap).
const LINK_END: usize = usize::MAX;

pub struct FreeListAllocator {
    heap_start: usize,
    heap_end: usize,
    /// Bump pointer, stored as an offset from `heap_start`: unallocated region starts here
    bump_next: core::sync::atomic::AtomicUsize,
    /// Free list head (protected by Mutex in test, UnsafeCell otherwise)
    #[cfg(test)]
    free_list: std::sync::Mutex<*mut FreeBlock>,
    #[cfg(not(test))]
    free_list: core::cell::UnsafeCell<*mut FreeBlock>,
}

#[cfg(test)]
unsafe impl Send for FreeListAllocator {}
#[cfg(test)]
unsafe impl Sync for FreeListAllocator {}
#[cfg(not(test))]
unsafe impl Send for FreeListAllocator {}
#[cfg(not(test))]
unsafe impl Sync for FreeListAllocator {}

impl FreeListAllocator {
    /// # Safety
    /// `heap_start..heap_end` must be a valid readable and writable memory region.
    pub unsafe fn new(heap_start: usize, heap_end: usize) -> Self {
        Self {
            heap_start,
            heap_end,
            bump_next: core::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            free_list: std::sync::Mutex::new(null_mut()),
            #[cfg(not(test))]
            free_list: core::cell::UnsafeCell::new(null_mut()),
        }
    }

    #[cfg(test)]
    fn free_list_head(&self) -> *mut FreeBlock {
        *self.free_list.lock().unwrap()
    }

    #[cfg(test)]
    fn set_free_list_head(&self, head: *mut FreeBlock) {
        *self.free_list.lock().unwrap() = head;
    }

    #[cfg(not(test))]
    fn free_list_head(&self) -> *mut FreeBlock {
        unsafe { *self.free_list.get() }
    }

    #[cfg(not(test))]
    fn set_free_list_head(&self, head: *mut FreeBlock) {
        unsafe { *self.free_list.get() = head }
    }

    /// Round `addr` up to a multiple of `align` (which must be a non-zero power of two).
    #[inline(always)]
    fn align_up(addr: usize, align: usize) -> usize {
        let mask = align - 1;
        (addr + mask) & !mask
    }

    /// Absolute address of a relative link, or `None` for "end of list".
    ///
    /// The sentinel is deliberately *not* offset 0: the very first byte of the heap is a
    /// legal allocation, so we use `usize::MAX` to mean "no successor".
    #[inline(always)]
    fn resolve(&self, rel: usize) -> Option<*mut FreeBlock> {
        if rel == LINK_END {
            None
        } else {
            Some((self.heap_start + rel) as *mut FreeBlock)
        }
    }

    #[inline(always)]
    fn relative(&self, ptr: *mut FreeBlock) -> usize {
        if ptr.is_null() {
            LINK_END
        } else {
            (ptr as usize).wrapping_sub(self.heap_start)
        }
    }

    /// Turn the memory at `addr` into a list node and return it. Returns `None` when the
    /// candidate block does not lie fully inside the heap or is too small to host a header.
    #[inline(always)]
    unsafe fn make_node(
        &self,
        addr: usize,
        size: usize,
        next: usize,
        bias: usize,
    ) -> Option<*mut FreeBlock> {
        if size < MIN_BLOCK || addr < self.heap_start || addr.saturating_add(size) > self.heap_end
        {
            return None;
        }
        let block = addr as *mut FreeBlock;
        block.write(FreeBlock { size, next, bias });
        Some(block)
    }
}

unsafe impl GlobalAlloc for FreeListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Ensure block is at least large enough to hold a FreeBlock header (for future dealloc)
        let size = layout.size().max(MIN_BLOCK);
        let align = layout
            .align()
            .max(core::mem::align_of::<FreeBlock>())
            .max(1);

        // ---- Step 1: first-fit over the free list --------------------------------------
        // `prev_slot` points at the predecessor's `next` field so unlinking is one store
        // regardless of position; for the head we use the list head accessor instead.
        let mut cursor = self.free_list_head();
        let mut prev_slot: *mut usize = null_mut();
        while !cursor.is_null() {
            let node = &mut *cursor;
            let (node_size, next_rel, bias) = (node.size, node.next, node.bias);
            let base = cursor as usize + bias;
            let aligned = Self::align_up(base, align);
            let slack = aligned - base; // bytes lost to alignment
            let usable = node_size - bias;

            if usable >= size && aligned.saturating_add(size) <= self.heap_end {
                // Unlink the chosen node first; any leftovers are re-linked below.
                if prev_slot.is_null() {
                    match self.resolve(next_rel) {
                        Some(head) => self.set_free_list_head(head),
                        None => self.set_free_list_head(null_mut()),
                    }
                } else {
                    *prev_slot = next_rel;
                }

                // Splitting policy: the returned object occupies [aligned, aligned+size),
                // so everything after it goes back on the list. A prefix that alignment
                // skipped is only worth re-linking when it can still host a header.
                let tail_size = usable - slack - size;
                if tail_size >= MIN_BLOCK {
                    self.push_front(aligned + size, tail_size, next_rel);
                } else if slack >= MIN_BLOCK {
                    // Nothing useful left behind the object; keep the skipped prefix alive.
                    self.push_front(base, slack, next_rel);
                }

                return aligned as *mut u8;
            }

            prev_slot = &mut (*cursor).next as *mut usize;
            cursor = match self.resolve((*cursor).next) {
                Some(n) => n,
                None => null_mut(),
            };
        }

        // ---- Step 2: fall back to the bump region -------------------------------------
        loop {
            // The cursor is an offset from heap_start, so a failed allocation can never
            // leave an out-of-range value behind.
            let offset = self.bump_next.load(Ordering::SeqCst);
            let aligned = Self::align_up(self.heap_start + offset, align);
            let new_offset = aligned.saturating_sub(self.heap_start) + size;
            if self.heap_start + new_offset > self.heap_end {
                return null_mut();
            }
            if self
                .bump_next
                .compare_exchange(offset, new_offset, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return aligned as *mut u8;
            }
            // Lost the race: another thread advanced the cursor, reload and retry.
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size().max(MIN_BLOCK);

        // Ignore pointers that were never carved out of this heap (defensive: dealloc of a
        // foreign/bump address would otherwise scribble over unrelated memory).
        let addr = ptr as usize;
        if addr < self.heap_start || addr.saturating_add(size) > self.heap_end {
            return;
        }

        // Head-insertion: write the intrusive header into the freed memory itself.
        let next_rel = self.relative(self.free_list_head());
        if let Some(block) = self.make_node(addr, size, next_rel, 0) {
            self.set_free_list_head(block);
        }
    }
}

impl FreeListAllocator {
    /// Build a node at `addr` covering `size` bytes and splice it in front of `rest_rel`,
    /// then publish it as the new head. Returns false when the region was rejected.
    #[inline]
    unsafe fn push_front(&self, addr: usize, size: usize, rest_rel: usize) -> bool {
        match self.make_node(addr, size, rest_rel, 0) {
            Some(block) => {
                self.set_free_list_head(block);
                true
            }
            None => false,
        }
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    const HEAP_SIZE: usize = 4096;

    fn make_allocator() -> (FreeListAllocator, Vec<u8>) {
        let mut heap = vec![0u8; HEAP_SIZE];
        let start = heap.as_mut_ptr() as usize;
        let alloc = unsafe { FreeListAllocator::new(start, start + HEAP_SIZE) };
        (alloc, heap)
    }

    #[test]
    fn test_alloc_basic() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(32, 8).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_alloc_alignment() {
        let (alloc, _heap) = make_allocator();
        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(8, align).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % align, 0, "align={align}");
        }
    }

    #[test]
    fn test_dealloc_and_reuse() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(64, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null());

        // After freeing, the next allocation should reuse the same block
        unsafe { alloc.dealloc(p1, layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        assert!(!p2.is_null());
        assert_eq!(p1, p2, "should reuse the freed block");
    }

    #[test]
    fn test_multiple_alloc_dealloc() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(128, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        let p3 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null() && !p2.is_null() && !p3.is_null());

        unsafe { alloc.dealloc(p2, layout) };
        unsafe { alloc.dealloc(p1, layout) };

        let q1 = unsafe { alloc.alloc(layout) };
        let q2 = unsafe { alloc.alloc(layout) };
        assert!(!q1.is_null() && !q2.is_null());
    }

    #[test]
    fn test_oom() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(HEAP_SIZE + 1, 1).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(ptr.is_null(), "should return null when exceeding heap");
    }
}
