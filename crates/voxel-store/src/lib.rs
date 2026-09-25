pub mod region;

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

pub const CHUNK_LEVELS: u8 = 3;
pub const CHUNK_SIZE: u32 = 4u32.pow(CHUNK_LEVELS as u32); // 64
pub const REGION_DIM: i64 = 16;
pub const SECTOR_SIZE: u32 = 4096;

pub const FORMAT_ABSENT: u8 = 0;
pub const FORMAT_HOT: u8 = 1;

pub type ChunkPos = [i64; 3];

pub fn chunk_to_region(pos: ChunkPos) -> [i64; 3] {
    pos.map(|c| c.div_euclid(REGION_DIM))
}

pub fn chunk_to_slot(pos: ChunkPos) -> usize {
    let local = pos.map(|c| c.rem_euclid(REGION_DIM) as usize);
    local[0] + local[1] * REGION_DIM as usize + local[2] * (REGION_DIM * REGION_DIM) as usize
}

pub fn region_filename(region_pos: [i64; 3]) -> String {
    format!(
        "r.{}.{}.{}.region",
        region_pos[0], region_pos[1], region_pos[2]
    )
}

#[derive(Debug)]
pub enum StorageError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u16),
    UnknownFormat(u8),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(e) => write!(f, "I/O error: {e}"),
            StorageError::InvalidMagic => write!(f, "invalid region file magic"),
            StorageError::UnsupportedVersion(v) => {
                write!(f, "unsupported region file version: {v}")
            }
            StorageError::UnknownFormat(tag) => write!(f, "unknown chunk format tag: {tag}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StorageError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for StorageError {
    fn from(e: io::Error) -> Self {
        StorageError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;

pub struct ChunkStore {
    dir: PathBuf,
    regions: HashMap<[i64; 3], region::RegionFile>,
}

impl ChunkStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            regions: HashMap::new(),
        })
    }

    pub fn save_chunk(&mut self, pos: ChunkPos, tree: &voxel_tree64::Tree64<u8>) -> Result<()> {
        let mut buf = Vec::new();
        tree.serialize(&mut buf).map_err(StorageError::Io)?;
        let region_pos = chunk_to_region(pos);
        let slot = chunk_to_slot(pos);
        let region = self.open_or_create_region(region_pos)?;
        region.write_chunk(slot, &buf, FORMAT_HOT)?;
        Ok(())
    }

    pub fn load_chunk(&mut self, pos: ChunkPos) -> Result<Option<voxel_tree64::Tree64<u8>>> {
        let region_pos = chunk_to_region(pos);
        let slot = chunk_to_slot(pos);
        let region = match self.try_open_region(region_pos)? {
            Some(r) => r,
            None => return Ok(None),
        };
        let data = match region.read_chunk(slot)? {
            Some((data, format_tag)) => {
                if format_tag != FORMAT_HOT {
                    return Err(StorageError::UnknownFormat(format_tag));
                }
                data
            }
            None => return Ok(None),
        };
        let tree = voxel_tree64::Tree64::deserialize(io::Cursor::new(data)).map_err(StorageError::Io)?;
        Ok(Some(tree))
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Result<()> {
        let region_pos = chunk_to_region(pos);
        let slot = chunk_to_slot(pos);
        if let Some(region) = self.try_open_region(region_pos)? {
            region.remove_chunk(slot)?;
        }
        Ok(())
    }

    pub fn has_chunk(&mut self, pos: ChunkPos) -> Result<bool> {
        let region_pos = chunk_to_region(pos);
        let slot = chunk_to_slot(pos);
        match self.try_open_region(region_pos)? {
            Some(region) => Ok(region.has_chunk(slot)),
            None => Ok(false),
        }
    }

    fn region_path(&self, region_pos: [i64; 3]) -> PathBuf {
        self.dir.join(region_filename(region_pos))
    }

    fn try_open_region(&mut self, region_pos: [i64; 3]) -> Result<Option<&mut region::RegionFile>> {
        if self.regions.contains_key(&region_pos) {
            return Ok(Some(self.regions.get_mut(&region_pos).unwrap()));
        }
        let path = self.region_path(region_pos);
        if !path.exists() {
            return Ok(None);
        }
        let rf = region::RegionFile::open(&path)?;
        self.regions.insert(region_pos, rf);
        Ok(Some(self.regions.get_mut(&region_pos).unwrap()))
    }

    fn open_or_create_region(&mut self, region_pos: [i64; 3]) -> Result<&mut region::RegionFile> {
        if !self.regions.contains_key(&region_pos) {
            let path = self.region_path(region_pos);
            let rf = region::RegionFile::open_or_create(&path)?;
            self.regions.insert(region_pos, rf);
        }
        Ok(self.regions.get_mut(&region_pos).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_to_region_positive() {
        assert_eq!(chunk_to_region([0, 0, 0]), [0, 0, 0]);
        assert_eq!(chunk_to_region([15, 15, 15]), [0, 0, 0]);
        assert_eq!(chunk_to_region([16, 0, 0]), [1, 0, 0]);
        assert_eq!(chunk_to_region([31, 31, 31]), [1, 1, 1]);
        assert_eq!(chunk_to_region([32, 32, 32]), [2, 2, 2]);
    }

    #[test]
    fn test_chunk_to_region_negative() {
        assert_eq!(chunk_to_region([-1, -1, -1]), [-1, -1, -1]);
        assert_eq!(chunk_to_region([-16, -16, -16]), [-1, -1, -1]);
        assert_eq!(chunk_to_region([-17, -17, -17]), [-2, -2, -2]);
    }

    #[test]
    fn test_chunk_to_slot() {
        assert_eq!(chunk_to_slot([0, 0, 0]), 0);
        assert_eq!(chunk_to_slot([1, 0, 0]), 1);
        assert_eq!(chunk_to_slot([0, 1, 0]), 16);
        assert_eq!(chunk_to_slot([0, 0, 1]), 256);
        assert_eq!(chunk_to_slot([15, 15, 15]), 4095);
    }

    #[test]
    fn test_chunk_to_slot_negative() {
        // -1 should map to local coord 15
        assert_eq!(chunk_to_slot([-1, 0, 0]), 15);
        assert_eq!(chunk_to_slot([0, -1, 0]), 15 * 16);
        assert_eq!(chunk_to_slot([-1, -1, -1]), 15 + 15 * 16 + 15 * 256);
    }

    #[test]
    fn test_region_filename() {
        assert_eq!(region_filename([0, 0, 0]), "r.0.0.0.region");
        assert_eq!(region_filename([1, -2, 3]), "r.1.-2.3.region");
    }

    #[test]
    fn test_chunk_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ChunkStore::new(dir.path()).unwrap();

        let values = [1_u8; 64];
        let tree = voxel_tree64::Tree64::new((&values[..], [4, 4, 4]));

        let pos: ChunkPos = [0, 0, 0];
        store.save_chunk(pos, &tree).unwrap();

        assert!(store.has_chunk(pos).unwrap());
        assert!(!store.has_chunk([1, 0, 0]).unwrap());

        let loaded = store.load_chunk(pos).unwrap().unwrap();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    assert_eq!(loaded.get_value_at([x, y, z]), tree.get_value_at([x, y, z]));
                }
            }
        }
    }

    #[test]
    fn test_chunk_store_negative_coords() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ChunkStore::new(dir.path()).unwrap();

        let values = [5_u8; 64];
        let tree = voxel_tree64::Tree64::new((&values[..], [4, 4, 4]));

        let pos: ChunkPos = [-10, -20, -30];
        store.save_chunk(pos, &tree).unwrap();

        let loaded = store.load_chunk(pos).unwrap().unwrap();
        assert_eq!(loaded.get_value_at([0, 0, 0]), Some(5));
    }

    #[test]
    fn test_chunk_store_multiple_regions() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ChunkStore::new(dir.path()).unwrap();

        let tree_a = voxel_tree64::Tree64::new((&[1_u8; 64][..], [4, 4, 4]));
        let tree_b = voxel_tree64::Tree64::new((&[2_u8; 64][..], [4, 4, 4]));

        let pos_a: ChunkPos = [0, 0, 0];
        let pos_b: ChunkPos = [16, 0, 0]; // different region
        store.save_chunk(pos_a, &tree_a).unwrap();
        store.save_chunk(pos_b, &tree_b).unwrap();

        let loaded_a = store.load_chunk(pos_a).unwrap().unwrap();
        let loaded_b = store.load_chunk(pos_b).unwrap().unwrap();
        assert_eq!(loaded_a.get_value_at([0, 0, 0]), Some(1));
        assert_eq!(loaded_b.get_value_at([0, 0, 0]), Some(2));
    }

    #[test]
    fn test_chunk_store_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ChunkStore::new(dir.path()).unwrap();
        assert!(store.load_chunk([0, 0, 0]).unwrap().is_none());
    }

    #[test]
    fn test_chunk_store_remove() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ChunkStore::new(dir.path()).unwrap();

        let tree = voxel_tree64::Tree64::new((&[1_u8; 64][..], [4, 4, 4]));
        let pos: ChunkPos = [0, 0, 0];

        store.save_chunk(pos, &tree).unwrap();
        assert!(store.has_chunk(pos).unwrap());

        store.remove_chunk(pos).unwrap();
        assert!(!store.has_chunk(pos).unwrap());
        assert!(store.load_chunk(pos).unwrap().is_none());
    }

    #[test]
    fn test_chunk_store_persistence() {
        let dir = tempfile::tempdir().unwrap();

        let tree = voxel_tree64::Tree64::new((&[3_u8; 64][..], [4, 4, 4]));
        let pos: ChunkPos = [5, 5, 5];

        {
            let mut store = ChunkStore::new(dir.path()).unwrap();
            store.save_chunk(pos, &tree).unwrap();
        }

        // Reopen from scratch
        let mut store = ChunkStore::new(dir.path()).unwrap();
        let loaded = store.load_chunk(pos).unwrap().unwrap();
        assert_eq!(loaded.get_value_at([0, 0, 0]), Some(3));
    }
}
