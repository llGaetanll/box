use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;

use crate::SECTOR_SIZE;
use crate::StorageError;

const MAGIC: &[u8; 4] = b"VXRG";
const VERSION: u16 = 1;
const FILE_HEADER_SIZE: usize = 8; // magic(4) + version(2) + reserved(2)
const ENTRY_SIZE: usize = 12;
const NUM_SLOTS: usize = 4096; // 16^3
const HEADER_TABLE_SIZE: usize = NUM_SLOTS * ENTRY_SIZE;
const HEADER_TOTAL_SIZE: usize = FILE_HEADER_SIZE + HEADER_TABLE_SIZE;
const DATA_START_SECTOR: u32 = sectors_needed(HEADER_TOTAL_SIZE as u32);

#[derive(Clone, Copy)]
struct HeaderEntry {
    sector_start: u32,
    byte_size: u32,
    format_tag: u8,
}

impl HeaderEntry {
    fn absent() -> Self {
        Self {
            sector_start: 0,
            byte_size: 0,
            format_tag: 0,
        }
    }

    fn is_present(&self) -> bool {
        self.byte_size > 0
    }

    fn sector_count(&self) -> u32 {
        sectors_needed(self.byte_size)
    }

    fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut buf = [0u8; ENTRY_SIZE];
        buf[0..4].copy_from_slice(&self.sector_start.to_le_bytes());
        buf[4..8].copy_from_slice(&self.byte_size.to_le_bytes());
        buf[8] = self.format_tag;
        // buf[9..12] reserved zeroes
        buf
    }

    fn from_bytes(buf: &[u8; ENTRY_SIZE]) -> Self {
        Self {
            sector_start: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            byte_size: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            format_tag: buf[8],
        }
    }
}

const fn sectors_needed(byte_size: u32) -> u32 {
    byte_size.div_ceil(SECTOR_SIZE)
}

pub struct RegionFile {
    file: File,
    entries: [HeaderEntry; NUM_SLOTS],
    next_sector: u32,
}

impl std::fmt::Debug for RegionFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionFile")
            .field("next_sector", &self.next_sector)
            .finish_non_exhaustive()
    }
}

impl RegionFile {
    pub fn create(path: &Path) -> crate::Result<Self> {
        let mut file = File::create(path)?;

        // Write file header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&[0u8; 2])?; // reserved

        // Write empty header table
        let empty_entry = HeaderEntry::absent().to_bytes();
        for _ in 0..NUM_SLOTS {
            file.write_all(&empty_entry)?;
        }

        // Pad to sector boundary
        let written = HEADER_TOTAL_SIZE;
        let padded = DATA_START_SECTOR as usize * SECTOR_SIZE as usize;
        if padded > written {
            let padding = vec![0u8; padded - written];
            file.write_all(&padding)?;
        }

        file.flush()?;

        // Reopen as read+write
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        Ok(Self {
            file,
            entries: [HeaderEntry::absent(); NUM_SLOTS],
            next_sector: DATA_START_SECTOR,
        })
    }

    pub fn open(path: &Path) -> crate::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        // Read and validate file header
        let mut header_buf = [0u8; FILE_HEADER_SIZE];
        file.read_exact(&mut header_buf)?;

        if &header_buf[0..4] != MAGIC {
            return Err(StorageError::InvalidMagic);
        }
        let version = u16::from_le_bytes([header_buf[4], header_buf[5]]);
        if version != VERSION {
            return Err(StorageError::UnsupportedVersion(version));
        }

        // Read header table
        let mut entries = [HeaderEntry::absent(); NUM_SLOTS];
        let mut entry_buf = [0u8; ENTRY_SIZE];
        for entry in &mut entries {
            file.read_exact(&mut entry_buf)?;
            *entry = HeaderEntry::from_bytes(&entry_buf);
        }

        // Compute next_sector from high-water mark
        let mut next_sector = DATA_START_SECTOR;
        for entry in &entries {
            if entry.is_present() {
                let end = entry.sector_start + entry.sector_count();
                if end > next_sector {
                    next_sector = end;
                }
            }
        }

        Ok(Self {
            file,
            entries,
            next_sector,
        })
    }

    pub fn open_or_create(path: &Path) -> crate::Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    pub fn has_chunk(&self, slot: usize) -> bool {
        self.entries[slot].is_present()
    }

    pub fn read_chunk(&mut self, slot: usize) -> crate::Result<Option<(Vec<u8>, u8)>> {
        let entry = self.entries[slot];
        if !entry.is_present() {
            return Ok(None);
        }

        let offset = entry.sector_start as u64 * SECTOR_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut data = vec![0u8; entry.byte_size as usize];
        self.file.read_exact(&mut data)?;

        Ok(Some((data, entry.format_tag)))
    }

    pub fn write_chunk(&mut self, slot: usize, data: &[u8], format_tag: u8) -> crate::Result<()> {
        let needed_sectors = sectors_needed(data.len() as u32);
        let old_entry = self.entries[slot];

        // Decide where to write: reuse old sectors if they fit, otherwise append
        let sector_start = if old_entry.is_present() && old_entry.sector_count() >= needed_sectors {
            old_entry.sector_start
        } else {
            let start = self.next_sector;
            self.next_sector += needed_sectors;
            start
        };

        // Write chunk data
        let offset = sector_start as u64 * SECTOR_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(data)?;

        // Zero-pad to sector boundary
        let remainder = data.len() % SECTOR_SIZE as usize;
        if remainder != 0 {
            let padding = vec![0u8; SECTOR_SIZE as usize - remainder];
            self.file.write_all(&padding)?;
        }

        // Update header entry in memory and on disk
        let new_entry = HeaderEntry {
            sector_start,
            byte_size: data.len() as u32,
            format_tag,
        };
        self.entries[slot] = new_entry;
        self.flush_entry(slot)?;

        Ok(())
    }

    pub fn remove_chunk(&mut self, slot: usize) -> crate::Result<()> {
        self.entries[slot] = HeaderEntry::absent();
        self.flush_entry(slot)?;
        Ok(())
    }

    fn flush_entry(&mut self, slot: usize) -> crate::Result<()> {
        let entry_offset = FILE_HEADER_SIZE as u64 + slot as u64 * ENTRY_SIZE as u64;
        self.file.seek(SeekFrom::Start(entry_offset))?;
        self.file.write_all(&self.entries[slot].to_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_entry_roundtrip() {
        let entry = HeaderEntry {
            sector_start: 42,
            byte_size: 12345,
            format_tag: 1,
        };
        let bytes = entry.to_bytes();
        let parsed = HeaderEntry::from_bytes(&bytes);
        assert_eq!(parsed.sector_start, 42);
        assert_eq!(parsed.byte_size, 12345);
        assert_eq!(parsed.format_tag, 1);
    }

    #[test]
    fn test_header_entry_absent() {
        let entry = HeaderEntry::absent();
        assert!(!entry.is_present());
        assert_eq!(entry.sector_start, 0);
        assert_eq!(entry.byte_size, 0);
        assert_eq!(entry.format_tag, 0);
    }

    #[test]
    fn test_sectors_needed() {
        assert_eq!(sectors_needed(0), 0);
        assert_eq!(sectors_needed(1), 1);
        assert_eq!(sectors_needed(4096), 1);
        assert_eq!(sectors_needed(4097), 2);
        assert_eq!(sectors_needed(8192), 2);
    }

    #[test]
    fn test_create_region_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let rf = RegionFile::create(&path).unwrap();
        assert_eq!(rf.next_sector, DATA_START_SECTOR);
        for slot in 0..NUM_SLOTS {
            assert!(!rf.has_chunk(slot));
        }

        // Verify file size
        let metadata = std::fs::metadata(&path).unwrap();
        let expected_size = DATA_START_SECTOR as u64 * SECTOR_SIZE as u64;
        assert_eq!(metadata.len(), expected_size);
    }

    #[test]
    fn test_open_validates_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.region");

        // Write a file with wrong magic but enough bytes for the full header
        let mut data = vec![0u8; HEADER_TOTAL_SIZE];
        data[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, &data).unwrap();

        let err = RegionFile::open(&path).unwrap_err();
        assert!(matches!(err, StorageError::InvalidMagic));
    }

    #[test]
    fn test_open_validates_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_ver.region");

        // Write a file with correct magic but wrong version
        let mut data = vec![0u8; HEADER_TOTAL_SIZE];
        data[0..4].copy_from_slice(MAGIC);
        data[4..6].copy_from_slice(&99u16.to_le_bytes());
        std::fs::write(&path, &data).unwrap();

        let err = RegionFile::open(&path).unwrap_err();
        assert!(matches!(err, StorageError::UnsupportedVersion(99)));
    }

    #[test]
    fn test_write_and_read_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();
        let data = vec![42u8; 100];
        rf.write_chunk(0, &data, 1).unwrap();

        assert!(rf.has_chunk(0));
        assert!(!rf.has_chunk(1));

        let (read_data, format_tag) = rf.read_chunk(0).unwrap().unwrap();
        assert_eq!(read_data, data);
        assert_eq!(format_tag, 1);
    }

    #[test]
    fn test_read_absent_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();
        assert!(rf.read_chunk(0).unwrap().is_none());
    }

    #[test]
    fn test_overwrite_smaller() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();

        let big = vec![1u8; 5000]; // 2 sectors
        rf.write_chunk(0, &big, 1).unwrap();
        let sector_before = rf.entries[0].sector_start;

        let small = vec![2u8; 100]; // 1 sector, fits in old 2
        rf.write_chunk(0, &small, 1).unwrap();
        let sector_after = rf.entries[0].sector_start;

        // Should reuse same sectors
        assert_eq!(sector_before, sector_after);

        let (read_data, _) = rf.read_chunk(0).unwrap().unwrap();
        assert_eq!(read_data, small);
    }

    #[test]
    fn test_overwrite_larger() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();

        let small = vec![1u8; 100]; // 1 sector
        rf.write_chunk(0, &small, 1).unwrap();
        let sector_before = rf.entries[0].sector_start;

        let big = vec![2u8; 5000]; // 2 sectors, doesn't fit
        rf.write_chunk(0, &big, 1).unwrap();
        let sector_after = rf.entries[0].sector_start;

        // Should allocate new sectors
        assert_ne!(sector_before, sector_after);

        let (read_data, _) = rf.read_chunk(0).unwrap().unwrap();
        assert_eq!(read_data, big);
    }

    #[test]
    fn test_remove_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();
        rf.write_chunk(0, &[1u8; 100], 1).unwrap();
        assert!(rf.has_chunk(0));

        rf.remove_chunk(0).unwrap();
        assert!(!rf.has_chunk(0));
        assert!(rf.read_chunk(0).unwrap().is_none());
    }

    #[test]
    fn test_multiple_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();

        for slot in 0..10 {
            let data = vec![slot as u8; 200 + slot * 100];
            rf.write_chunk(slot, &data, 1).unwrap();
        }

        for slot in 0..10 {
            let (data, _) = rf.read_chunk(slot).unwrap().unwrap();
            assert_eq!(data.len(), 200 + slot * 100);
            assert!(data.iter().all(|&b| b == slot as u8));
        }
    }

    #[test]
    fn test_persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        {
            let mut rf = RegionFile::create(&path).unwrap();
            rf.write_chunk(0, &[10u8; 500], 1).unwrap();
            rf.write_chunk(100, &[20u8; 3000], 1).unwrap();
        }

        let mut rf = RegionFile::open(&path).unwrap();
        assert!(rf.has_chunk(0));
        assert!(rf.has_chunk(100));
        assert!(!rf.has_chunk(1));

        let (data0, _) = rf.read_chunk(0).unwrap().unwrap();
        assert_eq!(data0, vec![10u8; 500]);

        let (data100, _) = rf.read_chunk(100).unwrap().unwrap();
        assert_eq!(data100, vec![20u8; 3000]);

        // Can still write after reopen
        rf.write_chunk(50, &[30u8; 100], 1).unwrap();
        let (data50, _) = rf.read_chunk(50).unwrap().unwrap();
        assert_eq!(data50, vec![30u8; 100]);
    }

    #[test]
    fn test_open_or_create() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        // Creates when doesn't exist
        let mut rf = RegionFile::open_or_create(&path).unwrap();
        rf.write_chunk(0, &[1u8; 100], 1).unwrap();
        drop(rf);

        // Opens when exists
        let mut rf = RegionFile::open_or_create(&path).unwrap();
        assert!(rf.has_chunk(0));
        let (data, _) = rf.read_chunk(0).unwrap().unwrap();
        assert_eq!(data, vec![1u8; 100]);
    }

    #[test]
    fn test_last_slot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.region");

        let mut rf = RegionFile::create(&path).unwrap();
        let last = NUM_SLOTS - 1;
        rf.write_chunk(last, &[99u8; 50], 1).unwrap();

        let (data, _) = rf.read_chunk(last).unwrap().unwrap();
        assert_eq!(data, vec![99u8; 50]);
    }
}
