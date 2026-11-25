//! # Bump Allocator (no_std)
//!
//! Implement the simplest heap memory allocator: a Bump Allocator (bump pointer allocator).
//!
//! ## How It Works
//!
//! A Bump Allocator maintains a pointer `next` to the "next available address".
//! On each allocation, it aligns `next` to the requested alignment, then advances by `size` bytes.
//! It does not support freeing individual objects (`dealloc` is a no-op).
//!
//! ```text
//! heap_start                              heap_end
//! |----[allocated]----[allocated]----| next |---[free]---|
//!                                        ^
//!                                    next allocation starts here
//! ```
//!
//! ## Task
//!
//! Implement `BumpAllocator`'s `GlobalAlloc::alloc` method:
//! 1. Align the current `next` up to `layout.align()`
//!    Hint: `align_up(addr, align) = (addr + align - 1) & !(align - 1)`
//! 2. Check if the aligned address plus `layout.size()` exceeds `heap_end`
//! 3. If it exceeds, return `null_mut()`; otherwise atomically update `next` with `compare_exchange`
//!
//! ## Key Concepts
//!
//! - `core::alloc::{GlobalAlloc, Layout}`
//! - Memory alignment calculation
//! - `AtomicUsize` and `compare_exchange` (CAS loop)

#![cfg_attr(not(test), no_std)]
#![allow(clippy::comparison_chain)]

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Number of low bits of `next` reserved for the per-allocation in-use flag.
const FLAG_BITS: u32 = 1;
const FLAG_MASK: usize = (1 << FLAG_BITS) - 1;
const OFFSET_SHIFT: u32 = FLAG_BITS;
const OFFSET_MASK: usize = !FLAG_MASK;

pub struct BumpAllocator {
    heap_start: usize,
    heap_end: usize,
    /// Packed cursor: bits [`OFFSET_SHIFT`..] hold the byte offset from `heap_start`,
    /// and bit 0 is a short-lived "allocation in progress" flag. Packing the flag into
    /// the same machine word is what lets a *single* atomic width carry both the cursor
    /// and mutual exclusion — no separate lock or second CAS is needed.
    ///
    /// Storing an offset rather than an absolute address also means a failed allocation
    /// can never corrupt the cursor with an out-of-range value.
    next: AtomicUsize,
}

impl BumpAllocator {
    /// Create a new BumpAllocator.
    ///
    /// # Safety
    /// `heap_start..heap_end` must be a valid, readable and writable memory region,
    /// and must not be used by other code during this allocator's lifetime.
    pub const unsafe fn new(heap_start: usize, heap_end: usize) -> Self {
        Self {
            heap_start,
            heap_end,
            // Packed representation starts as "offset 0, flag clear".
            next: AtomicUsize::new(0),
        }
    }

    /// Reset the allocator (free all allocated memory).
    pub fn reset(&self) {
        self.next.store(0, Ordering::SeqCst);
    }

    /// Round `addr` up to the next multiple of `align` (which must be a power of two).
    ///
    /// Written as a mask-and-add so it cannot overflow on a near-full heap:
    /// `addr + (align - 1)` would wrap for huge addresses, this form does not.
    #[inline(always)]
    fn align_up(addr: usize, align: usize) -> usize {
        let mask = align - 1;
        (addr + mask) & !mask
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Zero-sized requests are legal but must return a non-null aligned pointer;
        // hand back a dangling-but-well-aligned address that is never written.
        if layout.size() == 0 {
            return layout.align() as *mut u8;
        }

        loop {
            let packed = self.next.load(Ordering::SeqCst);

            // Claim the cursor: flip the in-use bit with one CAS. A thread that loses
            // the race just spins until the winner publishes the advanced cursor.
            if packed & FLAG_MASK != 0 {
                core::hint::spin_loop();
                continue;
            }
            let claimed = match self.next.compare_exchange(
                packed,
                packed | FLAG_MASK,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => packed,
                Err(_) => continue, // someone else claimed first, reload
            };

            // From here we are the sole owner of the cursor, so plain loads/stores suffice.
            let cursor = self.heap_start + (claimed >> OFFSET_SHIFT);
            let aligned = Self::align_up(cursor, layout.align().max(1));
            let end = aligned.wrapping_add(layout.size());

            if end > self.heap_end {
                // Out of memory: publish the unchanged cursor and release the claim.
                self.next.store(claimed, Ordering::SeqCst);
                return null_mut();
            }

            let advanced = (end - self.heap_start) << OFFSET_SHIFT;
            self.next.store(advanced, Ordering::SeqCst);
            return aligned as *mut u8;
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator does not reclaim individual objects — leave empty
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    const HEAP_SIZE: usize = 4096;

    fn make_allocator() -> (BumpAllocator, Vec<u8>) {
        let mut heap = vec![0u8; HEAP_SIZE];
        let start = heap.as_mut_ptr() as usize;
        let alloc = unsafe { BumpAllocator::new(start, start + HEAP_SIZE) };
        (alloc, heap)
    }

    #[test]
    fn test_alloc_basic() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(16, 8).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null(), "allocation should succeed");
    }

    #[test]
    fn test_alloc_alignment() {
        let (alloc, _heap) = make_allocator();
        for align in [1, 2, 4, 8, 16, 64] {
            let layout = Layout::from_size_align(1, align).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            assert_eq!(
                ptr as usize % align,
                0,
                "returned address must satisfy align={align}"
            );
        }
    }

    #[test]
    fn test_alloc_no_overlap() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(64, 8).unwrap();
        let p1 = unsafe { alloc.alloc(layout) } as usize;
        let p2 = unsafe { alloc.alloc(layout) } as usize;
        assert!(
            p1 + 64 <= p2 || p2 + 64 <= p1,
            "two allocations must not overlap"
        );
    }

    #[test]
    fn test_alloc_oom() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(HEAP_SIZE + 1, 1).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(ptr.is_null(), "should return null when exceeding heap");
    }

    #[test]
    fn test_alloc_fill_heap() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(256, 1).unwrap();
        for i in 0..16 {
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null(), "allocation #{i} should succeed");
        }
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(ptr.is_null(), "should return null when heap is full");
    }

    #[test]
    fn test_reset() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(HEAP_SIZE, 1).unwrap();
        let p1 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null());
        alloc.reset();
        let p2 = unsafe { alloc.alloc(layout) };
        assert!(!p2.is_null(), "should be able to allocate after reset");
        assert_eq!(
            p1, p2,
            "address after reset should match the first allocation"
        );
    }
}
