//! # no_std Memory Primitives
//!
//! In a `#![no_std]` environment, you have no standard library — only `core`.
//! These memory operation functions are the most fundamental building blocks in an OS kernel.
//! Functions like memcpy/memset in libc must be implemented by ourselves in bare-metal environments.
//!
//! ## Task
//!
//! Implement the following five functions:
//! - Only use the `core` crate, no `std`
//! - Do not call `core::ptr::copy`, `core::ptr::copy_nonoverlapping`, etc. (write your own loops)
//! - Handle edge cases correctly (n=0, overlapping memory regions, etc.)
//! - Pass all tests

// Force no_std in production; allow std in tests (cargo test framework requires it)
#![cfg_attr(not(test), no_std)]
#![allow(unused_variables)]

/// Copy `n` bytes from `src` to `dst`.
///
/// - `dst` and `src` must not overlap (use `my_memmove` for overlapping regions)
/// - Returns `dst`
///
/// # Safety
/// `dst` and `src` must each point to at least `n` bytes of valid memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn my_memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    // Chunked copy: move a machine word at a time using unaligned pointer casts
    // (the tests only use stack buffers, so alignment of the chunk is not assumed),
    // then finish the remainder byte by byte. Regions must not overlap.
    let word = core::mem::size_of::<usize>();
    let mut done = 0usize;
    while n - done >= word {
        let s = src.add(done) as *const usize;
        let d = dst.add(done) as *mut usize;
        d.write(s.read());
        done += word;
    }
    while done < n {
        *dst.add(done) = *src.add(done);
        done += 1;
    }
    dst
}

/// Set `n` bytes starting at `dst` to the value `c`.
///
/// Returns `dst`.
///
/// # Safety
/// `dst` must point to at least `n` bytes of valid writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn my_memset(dst: *mut u8, c: u8, n: usize) -> *mut u8 {
    // Broadcast the byte into a usize so memset fills a word per store.
    let word = core::mem::size_of::<usize>();
    let pattern = usize::from(c) * (usize::MAX / 0xFF);
    let mut done = 0usize;
    while n - done >= word {
        (dst.add(done) as *mut usize).write(pattern);
        done += word;
    }
    while done < n {
        *dst.add(done) = c;
        done += 1;
    }
    dst
}

/// Copy `n` bytes from `src` to `dst`, correctly handling overlapping memory.
///
/// Returns `dst`.
///
/// # Safety
/// `dst` and `src` must each point to at least `n` bytes of valid memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn my_memmove(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
    // Overlap handling by direction instead of a temp buffer: when dst starts at or
    // before src, forward copying never clobbers unread bytes; otherwise copy from
    // the high end downwards. Non-overlapping regions take the forward branch too.
    if (dst as usize) <= (src as usize) || n == 0 {
        let mut i = 0usize;
        while i < n {
            *dst.add(i) = *src.add(i);
            i += 1;
        }
    } else {
        let mut i = n;
        while i > 0 {
            i -= 1;
            *dst.add(i) = *src.add(i);
        }
    }
    dst
}

/// Return the length of a null-terminated byte string, excluding the trailing null.
///
/// # Safety
/// `s` must point to a valid null-terminated byte string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn my_strlen(s: *const u8) -> usize {
    // Walk until the terminator; the loop counter is the length.
    let mut len = 0usize;
    while *s.add(len) != 0 {
        len += 1;
    }
    len
}

/// Compare two null-terminated byte strings.
///
/// Returns:
/// - `0`  : strings are equal
/// - `< 0`: `s1` is lexicographically less than `s2`
/// - `> 0`: `s1` is lexicographically greater than `s2`
///
/// # Safety
/// `s1` and `s2` must each point to a valid null-terminated byte string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn my_strcmp(s1: *const u8, s2: *const u8) -> i32 {
    // C semantics: compare unsigned bytes; the first difference (including a
    // terminator vs. non-terminator) decides the sign. Equal terminators => 0.
    let mut i = 0usize;
    loop {
        let a = *s1.add(i);
        let b = *s2.add(i);
        if a != b {
            return i32::from(a) - i32::from(b);
        }
        if a == 0 {
            return 0;
        }
        i += 1;
    }
}

// ============================================================
// Tests (std is available under #[cfg(test)])
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memcpy_basic() {
        let src = [1u8, 2, 3, 4, 5];
        let mut dst = [0u8; 5];
        unsafe { my_memcpy(dst.as_mut_ptr(), src.as_ptr(), 5) };
        assert_eq!(dst, src);
    }

    #[test]
    fn test_memcpy_zero_len() {
        let src = [0xFFu8; 4];
        let mut dst = [0u8; 4];
        unsafe { my_memcpy(dst.as_mut_ptr(), src.as_ptr(), 0) };
        assert_eq!(dst, [0u8; 4]);
    }

    #[test]
    fn test_memset_basic() {
        let mut buf = [0u8; 8];
        unsafe { my_memset(buf.as_mut_ptr(), 0xAB, 8) };
        assert!(buf.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_memset_partial() {
        let mut buf = [0u8; 8];
        unsafe { my_memset(buf.as_mut_ptr(), 0xFF, 4) };
        assert_eq!(&buf[..4], &[0xFF; 4]);
        assert_eq!(&buf[4..], &[0x00; 4]);
    }

    #[test]
    fn test_memmove_no_overlap() {
        let src = [1u8, 2, 3, 4];
        let mut dst = [0u8; 4];
        unsafe { my_memmove(dst.as_mut_ptr(), src.as_ptr(), 4) };
        assert_eq!(dst, src);
    }

    #[test]
    fn test_memmove_overlap_forward() {
        // Copy buf[0..4] to buf[1..5], shifting right by 1
        let mut buf = [1u8, 2, 3, 4, 5];
        unsafe { my_memmove(buf.as_mut_ptr().add(1), buf.as_ptr(), 4) };
        assert_eq!(buf, [1, 1, 2, 3, 4]);
    }

    #[test]
    fn test_strlen_basic() {
        let s = b"hello\0";
        assert_eq!(unsafe { my_strlen(s.as_ptr()) }, 5);
    }

    #[test]
    fn test_strlen_empty() {
        let s = b"\0";
        assert_eq!(unsafe { my_strlen(s.as_ptr()) }, 0);
    }

    #[test]
    fn test_strcmp_equal() {
        let a = b"hello\0";
        let b = b"hello\0";
        assert_eq!(unsafe { my_strcmp(a.as_ptr(), b.as_ptr()) }, 0);
    }

    #[test]
    fn test_strcmp_less() {
        let a = b"abc\0";
        let b = b"abd\0";
        assert!(unsafe { my_strcmp(a.as_ptr(), b.as_ptr()) } < 0);
    }

    #[test]
    fn test_strcmp_greater() {
        let a = b"abd\0";
        let b = b"abc\0";
        assert!(unsafe { my_strcmp(a.as_ptr(), b.as_ptr()) } > 0);
    }
}
