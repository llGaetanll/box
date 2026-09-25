use std::fmt::Debug;
use std::io;

fn count_ones_variable(value: u64, index: u32) -> u32 {
    (value & ((1 << index) - 1)).count_ones()
}

fn set_bit(mask: u64, index: u32, value: bool) -> u64 {
    (mask & !(1 << index)) | ((value as u64) << index)
}

#[repr(C, packed)]
#[derive(
    Default, Debug, Clone, Copy, std::hash::Hash, PartialEq, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Node {
    pub is_leaf_and_ptr: u32,
    pub pop_mask: u64,
}

impl Node {
    fn empty(is_leaf: bool) -> Self {
        Self::new(is_leaf, 0, 0)
    }

    pub fn new(is_leaf: bool, ptr: u32, pop_mask: u64) -> Self {
        Self {
            is_leaf_and_ptr: (ptr << 1) | (is_leaf as u32),
            pop_mask,
        }
    }

    fn is_leaf(&self) -> bool {
        (self.is_leaf_and_ptr & 1) == 1
    }

    fn ptr(&self) -> u32 {
        self.is_leaf_and_ptr >> 1
    }

    fn is_occupied(&self, index: u32) -> bool {
        self.pop_mask >> index & 1 == 1
    }

    fn get_index_for_child(&self, child: u32) -> Option<u32> {
        Some(self.ptr() + self.get_index_in_children(child)?)
    }

    fn get_index_in_children(&self, index: u32) -> Option<u32> {
        if !self.is_occupied(index) {
            return None;
        }
        Some(count_ones_variable(self.pop_mask, index))
    }

    fn range(&self) -> std::ops::Range<usize> {
        self.ptr() as usize..self.ptr() as usize + self.pop_mask.count_ones() as usize
    }

    fn check_empty(&self) -> Option<Self> {
        Some(*self).filter(|node| node.pop_mask != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundingBox {
    min: glam::UVec3,
    max: glam::UVec3,
}

impl BoundingBox {
    fn from_levels(num_levels: u8) -> Self {
        Self {
            min: glam::UVec3::splat(0),
            max: glam::UVec3::splat(4_u32.pow(num_levels as u32)),
        }
    }

    fn get_intersection(&self, other: &Self) -> Option<BoundingBox> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min.cmplt(max).all() {
            Some(BoundingBox { min, max })
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct PopMaskedData<T: Default + Copy> {
    pub array: [T; 64],
    pub pop_mask: u64,
}

impl<T: Copy + Default> Default for PopMaskedData<T> {
    fn default() -> Self {
        Self {
            array: [T::default(); 64],
            pop_mask: 0,
        }
    }
}

impl<T: Copy + Default> PopMaskedData<T> {
    fn new(slice: &[T], pop_mask: u64) -> Self {
        let mut flat = [T::default(); 64];
        let mut var_pop_mask = pop_mask;

        for &value in slice {
            let next_pos = var_pop_mask.trailing_zeros();
            flat[next_pos as usize] = value;
            var_pop_mask &= var_pop_mask - 1;
        }

        Self {
            pop_mask,
            array: flat,
        }
    }

    pub fn as_compact(&self) -> arrayvec::ArrayVec<T, 64> {
        let mut array = arrayvec::ArrayVec::new();
        let mut var_pop_mask = self.pop_mask;

        while var_pop_mask != 0 {
            let next_pos = var_pop_mask.trailing_zeros();
            array.push(self.array[next_pos as usize]);
            var_pop_mask &= var_pop_mask - 1;
        }
        array
    }

    pub fn set(&mut self, index: u32, value: Option<T>) {
        self.array[index as usize] = value.unwrap_or_default();
        self.pop_mask = set_bit(self.pop_mask, index, value.is_some());
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct RootState {
    pub index: u32,
    pub num_levels: u8,
    pub offset: glam::IVec3,
}

#[derive(Debug, Default, PartialEq)]
pub struct Edits {
    root_states: Vec<RootState>,
    index: u32,
}

impl Edits {
    fn push(&mut self, state: RootState) {
        if !self.root_states.is_empty() {
            if state == self.current() {
                return;
            }
            while self.can_redo() {
                self.root_states.pop();
            }
        }
        self.index = self.root_states.len() as u32;
        self.root_states.push(state);
    }

    fn current(&self) -> RootState {
        self.root_states[self.index as usize]
    }

    pub fn can_undo(&self) -> bool {
        self.index > 0
    }

    pub fn can_redo(&self) -> bool {
        self.index < (self.root_states.len() as u32 - 1)
    }

    pub fn undo(&mut self) {
        if self.can_undo() {
            self.index -= 1;
        }
    }

    pub fn redo(&mut self) {
        if self.can_redo() {
            self.index += 1;
        }
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct UpdatedRanges {
    pub nodes: std::ops::Range<usize>,
    pub data: std::ops::Range<usize>,
}

pub trait VoxelModel<T> {
    fn dimensions(&self) -> [u32; 3];
    fn access(&self, coord: [usize; 3]) -> Option<T>;
}

pub struct FlatArray<'a, T> {
    pub values: &'a [T],
    pub dimensions: [u32; 3],
    pub empty_value: T,
}

impl<T: PartialEq + Clone + Default> VoxelModel<T> for FlatArray<'_, T> {
    fn dimensions(&self) -> [u32; 3] {
        self.dimensions
    }

    fn access(&self, [x, y, z]: [usize; 3]) -> Option<T> {
        if x >= self.dimensions[0] as usize
            || y >= self.dimensions[1] as usize
            || z >= self.dimensions[2] as usize
        {
            return None;
        }
        self.values
            .get(
                x + y * self.dimensions[0] as usize
                    + z * self.dimensions[0] as usize * self.dimensions[1] as usize,
            )
            .cloned()
            .filter(|value| *value != self.empty_value)
    }
}

impl<T: PartialEq + Clone + Default> VoxelModel<T> for (&'_ [T], [u32; 3]) {
    fn dimensions(&self) -> [u32; 3] {
        FlatArray {
            values: self.0,
            dimensions: self.1,
            empty_value: T::default(),
        }
        .dimensions()
    }

    fn access(&self, coord: [usize; 3]) -> Option<T> {
        FlatArray {
            values: self.0,
            dimensions: self.1,
            empty_value: T::default(),
        }
        .access(coord)
    }
}

impl<T: PartialEq + Clone + Default, M: VoxelModel<T>> VoxelModel<T> for &'_ M {
    fn dimensions(&self) -> [u32; 3] {
        M::dimensions(self)
    }

    fn access(&self, coord: [usize; 3]) -> Option<T> {
        M::access(self, coord)
    }
}

#[derive(Debug)]
pub struct Tree64<T> {
    pub nodes: Vec<Node>,
    pub data: Vec<T>,
    pub edits: Edits,
}

impl<
    T: std::hash::Hash
        + Clone
        + Copy
        + Default
        + PartialEq
        + Debug
        + bytemuck::Pod
        + bytemuck::Zeroable,
> Tree64<T>
{
    pub fn insert_values(&mut self, values: &[T]) -> u32 {
        let index = self.data.len();
        self.data.extend_from_slice(values);
        index as _
    }

    pub fn insert_nodes(&mut self, nodes: &[Node]) -> u32 {
        let index = self.nodes.len();
        self.nodes.extend_from_slice(nodes);
        index as _
    }

    pub fn root_state(&self) -> RootState {
        self.edits.current()
    }

    pub fn push_new_root_node(&mut self, node: Node, num_levels: u8, offset: glam::IVec3) -> u32 {
        let index = self.insert_nodes(&[node]);
        self.edits.push(RootState {
            index,
            num_levels,
            offset,
        });
        index
    }

    pub fn expand(&mut self) -> std::ops::Range<usize> {
        let root_state = self.root_state();
        let current_len = self.nodes.len();
        let new_nodes = [self.nodes[root_state.index as usize]; 64];
        let children = self.insert_nodes(&new_nodes);
        self.push_new_root_node(
            Node::new(false, children, !0),
            root_state.num_levels + 1,
            root_state.offset,
        );
        current_len..self.nodes.len()
    }

    pub fn new<M: VoxelModel<T>>(model: M) -> Self {
        let dims = model.dimensions();
        let mut scale = dims[0].max(dims[1]).max(dims[2]).next_power_of_two();
        scale = scale.max(4);
        if scale.ilog2() % 2 == 1 {
            scale *= 2;
        }
        let num_levels = scale.ilog(4) as _;

        let mut this = Self {
            nodes: Vec::new(),
            data: Vec::new(),
            edits: Edits::default(),
        };

        let root = this.insert_from_model_recursive(&model, glam::UVec3::ZERO, num_levels);
        let root_index = this.insert_nodes(&[root]);
        this.edits.push(RootState {
            index: root_index,
            num_levels,
            offset: glam::IVec3::ZERO,
        });
        this
    }

    fn insert_from_model_recursive<M: VoxelModel<T>>(
        &mut self,
        model: &M,
        offset: glam::UVec3,
        node_level: u8,
    ) -> Node {
        let mut bitmask = 0;

        if node_level == 1 {
            let mut vec = arrayvec::ArrayVec::<_, 64>::new();
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let pos = glam::UVec3::new(x, y, z);
                        let index = offset + pos;
                        if let Some(value) =
                            model.access([index.x as _, index.y as _, index.z as _])
                        {
                            vec.push(value);
                            bitmask |= 1 << pos.dot(glam::UVec3::new(1, 4, 16)) as u64;
                        }
                    }
                }
            }
            Node::new(true, self.insert_values(&vec), bitmask)
        } else {
            let new_scale = 4_u32.pow(node_level as u32 - 1);
            let mut nodes = arrayvec::ArrayVec::<_, 64>::new();
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let pos = glam::UVec3::new(x, y, z);
                        if let Some(child) = self
                            .insert_from_model_recursive(
                                model,
                                offset + pos * new_scale,
                                node_level - 1,
                            )
                            .check_empty()
                        {
                            nodes.push(child);
                            bitmask |= 1 << pos.dot(glam::UVec3::new(1, 4, 16)) as u64;
                        }
                    }
                }
            }
            Node::new(false, self.insert_nodes(&nodes), bitmask)
        }
    }

    pub fn get_value_at(&self, pos: [u32; 3]) -> Option<T> {
        let pos = glam::UVec3::from(pos);
        let state = self.root_state();
        let mut node = self.nodes[state.index as usize];
        let mut child_size = 4_u32.pow(state.num_levels as u32 - 1);

        while !node.is_leaf() {
            let child_index = ((pos / child_size) % 4).dot(glam::UVec3::new(1, 4, 16));
            child_size /= 4;
            if let Some(child_node_index) = node.get_index_for_child(child_index) {
                node = self.nodes[child_node_index as usize];
            } else {
                return None;
            }
        }

        debug_assert_eq!(child_size, 1);

        let child_index = (pos % 4).dot(glam::UVec3::new(1, 4, 16));
        node.get_index_for_child(child_index)
            .map(|index| self.data[index as usize])
    }

    pub fn modify(&mut self, at: [i32; 3], value: Option<T>) -> UpdatedRanges {
        self.modify_nodes_in_box(at, [at[0] + 1, at[1] + 1, at[2] + 1], value)
    }

    pub fn modify_nodes_in_box(
        &mut self,
        min: [i32; 3],
        max: [i32; 3],
        value: Option<T>,
    ) -> UpdatedRanges {
        let num_data = self.data.len();
        let num_nodes = self.nodes.len();

        let mut root_state = self.root_state();

        let mut min = glam::IVec3::from(min) + root_state.offset;
        let mut max = glam::IVec3::from(max) + root_state.offset;

        if min.cmpge(max).any() {
            return UpdatedRanges {
                data: num_data..self.data.len(),
                nodes: num_nodes..self.nodes.len(),
            };
        }

        let mut bbox = BoundingBox {
            min: min.as_uvec3(),
            max: max.as_uvec3(),
        };

        while bbox
            .get_intersection(&BoundingBox::from_levels(root_state.num_levels))
            .map(|intersection| intersection != bbox)
            .unwrap_or(true)
        {
            let index_of_existing_root = 1 + 4 + 16;
            root_state.index = self.insert_nodes(&[Node::new(
                false,
                root_state.index,
                1 << index_of_existing_root,
            )]);

            let offset = glam::IVec3::splat(4_i32.pow(root_state.num_levels as u32));

            min += offset;
            max += offset;

            bbox = BoundingBox {
                min: min.as_uvec3(),
                max: max.as_uvec3(),
            };

            root_state.num_levels += 1;
            root_state.offset += offset;
        }

        let new_root_node =
            self.modify_recursive(root_state.num_levels, Some(root_state.index), bbox, value);
        self.push_new_root_node(new_root_node, root_state.num_levels, root_state.offset);

        UpdatedRanges {
            data: num_data..self.data.len(),
            nodes: num_nodes..self.nodes.len(),
        }
    }

    fn modify_recursive(
        &mut self,
        level: u8,
        corresponding_node_index: Option<u32>,
        bbox: BoundingBox,
        value: Option<T>,
    ) -> Node {
        let is_leaf = level == 1;

        let node = if let Some(index) = corresponding_node_index {
            self.nodes[index as usize]
        } else {
            Node::empty(is_leaf)
        };

        if corresponding_node_index.is_none() && value.is_none() {
            return node;
        }

        let size = 4_u32.pow(level as u32);
        let node_bbox = BoundingBox {
            min: glam::UVec3::ZERO,
            max: glam::UVec3::splat(size),
        };

        let intersection = match node_bbox.get_intersection(&bbox) {
            None => return node,
            Some(intersection) => intersection,
        };

        if intersection == node_bbox {
            return if let Some(value) = value {
                let mut node = Node::new(true, self.insert_values(&[value; 64]), !0);
                for _ in 0..level.saturating_sub(1) {
                    node = Node::new(false, self.insert_nodes(&[node; 64]), !0);
                }
                node
            } else {
                Node::empty(is_leaf)
            };
        }

        if is_leaf {
            let mut pop_masked_data = PopMaskedData::new(&self.data[node.range()], node.pop_mask);

            for x in intersection.min.x..intersection.max.x {
                for y in intersection.min.y..intersection.max.y {
                    for z in intersection.min.z..intersection.max.z {
                        let index = glam::UVec3::new(x, y, z).dot(glam::UVec3::new(1, 4, 16));
                        pop_masked_data.set(index, value);
                    }
                }
            }

            Node::new(
                true,
                self.insert_values(&pop_masked_data.as_compact()),
                pop_masked_data.pop_mask,
            )
        } else {
            let mut pop_masked_data = PopMaskedData::new(&self.nodes[node.range()], node.pop_mask);
            let child_size = size / 4;

            let node_intersection = BoundingBox {
                min: intersection.min / child_size,
                max: (intersection.max + child_size - 1) / child_size,
            };

            for x in node_intersection.min.x..node_intersection.max.x {
                for y in node_intersection.min.y..node_intersection.max.y {
                    for z in node_intersection.min.z..node_intersection.max.z {
                        let pos = glam::UVec3::new(x, y, z);
                        let index = pos.dot(glam::UVec3::new(1, 4, 16));
                        let child_intersection = BoundingBox {
                            min: intersection.min.saturating_sub(child_size * pos),
                            max: intersection.max.saturating_sub(child_size * pos),
                        };
                        pop_masked_data.set(
                            index,
                            Some(self.modify_recursive(
                                level - 1,
                                node.get_index_for_child(index),
                                child_intersection,
                                value,
                            ))
                            .filter(|node| node.pop_mask != 0),
                        );
                    }
                }
            }

            Node::new(
                false,
                self.insert_nodes(&pop_masked_data.as_compact()),
                pop_masked_data.pop_mask,
            )
        }
    }

    pub fn serialize<W: io::Write>(&self, mut writer: W) -> io::Result<()> {
        let root_state = self.root_state();
        writer.write_all(&[root_state.num_levels])?;
        writer.write_all(&root_state.index.to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&root_state.offset.to_array()))?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.data.len() as u32).to_le_bytes())?;
        writer.write_all(bytemuck::cast_slice(&self.nodes))?;
        writer.write_all(bytemuck::cast_slice(&self.data))?;
        Ok(())
    }

    pub fn deserialize<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut num_levels = 0;
        let mut num_nodes = 0_u32;
        let mut num_data = 0_u32;
        let mut root_node_index = 0_u32;
        let mut root_offset = [0_i32; 3];
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_levels))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut root_node_index))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut root_offset))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_nodes))?;
        reader.read_exact(bytemuck::bytes_of_mut(&mut num_data))?;
        let mut this = Self {
            nodes: vec![Node::default(); num_nodes as usize],
            data: vec![T::default(); num_data as usize],
            edits: Edits {
                root_states: vec![RootState {
                    num_levels,
                    offset: root_offset.into(),
                    index: root_node_index,
                }],
                index: 0,
            },
        };
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.nodes))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut this.data))?;
        Ok(this)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_size() {
        assert_eq!(std::mem::size_of::<Node>(), 4 + 8);
    }

    #[test]
    fn test_pm() {
        let mut x = PopMaskedData::<u8>::default();
        for i in 0..64 {
            x.set(i, Some(1));
        }
        assert_eq!(&x.as_compact()[..], &[1; 64]);
        for i in (0..64).rev() {
            x.set(i, Some(2));
        }
        assert_eq!(&x.as_compact()[..], &[2; 64]);

        let mut x = PopMaskedData::<u8>::default();
        for i in (0..64).rev() {
            x.set(i, Some(3));
        }
        assert_eq!(&x.as_compact()[..], &[3; 64]);
        for i in (0..64).rev() {
            x.set(i, None);
        }
        let empty_slice: &[u8] = &[];
        assert_eq!(&*x.as_compact(), empty_slice);

        let mut x = PopMaskedData::<u8>::default();
        for i in 0..64 {
            x.set(i, Some(1));
        }
        x.set(1, None);
        assert_eq!(&x.as_compact()[..], &[1; 63]);

        let mut x = PopMaskedData::<u8>::new(&[1_u8; 4], 0b11110);
        x.set(0, Some(255));
        assert_eq!(&x.as_compact()[..], &[255, 1, 1, 1, 1]);
    }

    #[test]
    fn test_tree() {
        {
            let tree = Tree64::new((&[1, 1, 1, 1][..], [2, 2, 1]));
            assert_eq!(&*tree.data, &[1; 4]);
            assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
            assert_eq!({ tree.nodes[0].pop_mask }, 0b00110011);

            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if z == 1 { None } else { Some(1) }
                        );
                    }
                }
            }
        }

        {
            let array = &[1, 1, 1, 1, 1, 0, 1, 1];
            let tree = Tree64::new((&array[..], [2_u32, 2, 2]));
            assert_eq!(&*tree.data, &[1; 7]);
            assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
            assert_eq!(
                { tree.nodes[0].pop_mask },
                0b00000000001100010000000000110011,
                "{:064b}",
                { tree.nodes[0].pop_mask }
            );

            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            Some(array[x as usize + y as usize * 2 + z as usize * 4])
                                .filter(|v| *v != 0)
                        );
                    }
                }
            }
        }

        {
            let mut values = [1; 64];
            values[63] = 0;

            let tree = Tree64::new((&values[..], [4, 4, 4]));
            assert_eq!(&*tree.data, &[1; 63]);
            assert_eq!({ tree.nodes[0].is_leaf_and_ptr }, 1);
            assert_eq!({ tree.nodes[0].pop_mask }, (!0 & !(1 << 63)));

            for x in 0..4 {
                for y in 0..4 {
                    for z in 0..4 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if x == 3 && y == 3 && z == 3 {
                                None
                            } else {
                                Some(1)
                            }
                        );
                    }
                }
            }
        }

        // Serialize/deserialize roundtrip
        {
            let mut values = [1; 64];
            values[63] = 0;
            let tree = Tree64::new((&values[..], [4, 4, 4]));

            let mut data = Vec::new();
            tree.serialize(&mut data).unwrap();
            let tree2 = Tree64::<u8>::deserialize(io::Cursor::new(&data)).unwrap();
            assert_eq!(&*tree.data, &*tree2.data);
            assert_eq!(&*tree.nodes, &*tree2.nodes);
            assert_eq!(tree.edits, tree2.edits);

            let mut tree = tree;
            let ranges = tree.modify([2, 2, 3], None);
            assert_eq!(ranges.nodes, 1..2);

            let mut tree = tree;
            let ranges = tree.modify([0, 0, 0], Some(2));
            assert_eq!(ranges.nodes, 2..3);
            assert_eq!(ranges.data.len(), 64 - 2);
        }
    }

    #[test]
    fn single_node_modifications() {
        {
            let values = [1; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify([3, 3, 3], None);
            assert_eq!(ranges.nodes.len(), 1);
        }

        {
            let values = [1; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify([3, 3, 3], Some(2));
            assert_eq!(ranges.nodes.len(), 1);
        }

        {
            let values = vec![1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));
            let updated_nodes = 64 + 1;

            let ranges = tree.modify([5, 5, 5], None);
            assert_eq!(ranges.nodes.len(), updated_nodes);

            for x in 0..16 {
                for y in 0..16 {
                    for z in 0..16 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if x == 5 && y == 5 && z == 5 {
                                None
                            } else {
                                Some(1)
                            }
                        );
                    }
                }
            }
        }

        {
            let values = vec![1_u16; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));
            let updated_nodes = 64 + 64 + 1;

            let ranges = tree.modify([63, 63, 63], None);
            assert_eq!(ranges.nodes.len(), updated_nodes);
        }

        {
            let values = vec![1_u32; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));
            let updated_nodes = 64 + 64 + 1;

            let ranges = tree.modify([63, 63, 63], Some(2));
            assert_eq!(ranges.nodes.len(), updated_nodes);
        }

        {
            let values = vec![1_u64; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));
            let updated_nodes = 64 + 64 + 1;

            let ranges = tree.modify([0, 0, 0], Some(2));
            assert_eq!(ranges.data.len(), 64);
            assert_eq!(ranges.nodes.len(), updated_nodes);
        }
    }

    #[test]
    fn modifications_in_box() {
        {
            let values = [1; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([1; 3], [2; 3], None);
            assert_eq!(ranges.nodes.len(), 1);
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([1; 3], [3; 3], None);
            assert_eq!(ranges.nodes.len(), 64 + 1);
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], Some(2));
            assert_eq!(ranges.nodes.len(), 64 + 1);
            assert_eq!(ranges.data.len(), 64);
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], Some(2));
            assert_eq!(ranges.data.len(), 64);
            assert_eq!(ranges.nodes.len(), 64 + 1);
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [4; 3], Some(2));
            assert_eq!(ranges.nodes.len(), 64 + 1);
        }
    }

    #[test]
    fn advanced_modifications_in_box() {
        {
            let values = vec![1_u128; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], Some(2));
            assert_eq!(ranges.data.len(), 64);
            assert_ne!(ranges.nodes.len(), 64 + 1);
        }

        {
            let values = vec![1_u32; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [3; 3], Some(2));
            assert_eq!(ranges.data.len(), 64);
            assert_ne!(ranges.nodes.len(), 64 + 1);
        }
    }

    #[test]
    fn modification_rec() {
        {
            let values = [0; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [4; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..64,
                    nodes: 1..2
                }
            );
        }

        {
            let values = [1; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [4; 3], None);
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 64..64,
                    nodes: 1..2
                }
            );
        }

        {
            let values = [0; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..64,
                    nodes: 1..1 + 64 + 1
                }
            );
            assert_eq!(
                { tree.nodes[tree.root_state().index as usize].pop_mask },
                !0
            );
        }

        {
            let values = [0; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([16; 3], [16; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..0,
                    nodes: 1..1
                }
            );
        }

        {
            let values = [0; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [1; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..1,
                    nodes: 1..2
                }
            );
        }

        {
            let values = [0; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([3; 3], [4; 3], Some(2));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..1,
                    nodes: 1..2
                }
            );
        }

        {
            let values = [0; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([4; 3], [5; 3], Some(2));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..1,
                    nodes: 1..3
                }
            );
        }
    }

    #[test]
    fn modifications_on_empty_spaces() {
        {
            let values = [0_u16; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify([0; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    nodes: 1..3,
                    data: 0..1,
                }
            );
        }

        {
            let values = vec![0_i128; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            let ranges = tree.modify([0; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    nodes: 1..4,
                    data: 0..1,
                }
            );
        }

        {
            let values = [0_u32; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    nodes: 1..1 + 64 + 1 + 1,
                    data: 0..64,
                }
            );
        }

        {
            let values = [0; 4 * 4 * 4];
            let mut tree = Tree64::new((&values[..], [4; 3]));

            let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..8,
                    nodes: 1..2,
                }
            );
        }

        {
            let values = [0; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([2; 3], [4; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..8,
                    nodes: 1..3,
                }
            );
        }

        {
            let mut tree = Tree64::new((&[0; 16 * 16 * 16][..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([6; 3], [8; 3], Some(1));
            assert_eq!(
                ranges,
                UpdatedRanges {
                    data: 0..8,
                    nodes: 1..3,
                }
            );
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([1; 3], [3; 3], None);
            assert_eq!(ranges.nodes.len(), 64 + 1);
        }

        {
            let values = [1; 16 * 16 * 16];
            let mut tree = Tree64::new((&values[..], [16; 3]));

            let ranges = tree.modify_nodes_in_box([0; 3], [16; 3], None);
            assert_eq!(ranges.nodes.len(), 1);
            assert!(ranges.data.is_empty());
        }
    }

    #[test]
    fn outside_modifications() {
        let empty_slice: &[u8] = &[];

        let mut tree = Tree64::new((empty_slice, [0; 3]));
        assert_eq!(
            tree.modify_nodes_in_box([64; 3], [65; 3], Some(1)),
            UpdatedRanges {
                nodes: 1..9,
                data: 0..1
            }
        );
    }

    #[test]
    fn modify_box_weird_sizes() {
        {
            let values = [0_u8; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            tree.modify([10; 3], Some(1));

            for x in 0..64 {
                for y in 0..64 {
                    for z in 0..64 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if x == 10 && y == 10 && z == 10 {
                                Some(1)
                            } else {
                                None
                            },
                            "{} {} {}",
                            x,
                            y,
                            z
                        );
                    }
                }
            }
        }

        {
            let values = [0_u8; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            tree.modify_nodes_in_box([0; 3], [20; 3], Some(1));

            for x in 0..64 {
                for y in 0..64 {
                    for z in 0..64 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if x < 20 && y < 20 && z < 20 {
                                Some(1)
                            } else {
                                None
                            },
                            "{:?}",
                            (x, y, z)
                        );
                    }
                }
            }
        }

        for start in 0..44 {
            let values = [0_u8; 64 * 64 * 64];
            let mut tree = Tree64::new((&values[..], [64; 3]));

            tree.modify_nodes_in_box([start; 3], [start + 20, start + 20, start + 20], Some(1));

            let start = start as u32;

            for x in 0..64 {
                for y in 0..64 {
                    for z in 0..64 {
                        assert_eq!(
                            tree.get_value_at([x, y, z]),
                            if x >= start
                                && x < (start + 20)
                                && y >= start
                                && y < (start + 20)
                                && z >= start
                                && z < (start + 20)
                            {
                                Some(1)
                            } else {
                                None
                            },
                            "{:?} {:?}",
                            (x, y, z),
                            start
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_tiny() {
        Tree64::new((&[0][..], [1; 3]));
    }

    #[test]
    fn test_undo_redo() {
        let values = [1; 4 * 4 * 4];
        let mut tree = Tree64::new((&values[..], [4; 3]));

        assert_eq!(tree.get_value_at([0, 0, 0]), Some(1));
        assert!(!tree.edits.can_undo());

        tree.modify([0, 0, 0], Some(2));
        assert_eq!(tree.get_value_at([0, 0, 0]), Some(2));
        assert!(tree.edits.can_undo());

        tree.edits.undo();
        assert_eq!(tree.get_value_at([0, 0, 0]), Some(1));
        assert!(tree.edits.can_redo());

        tree.edits.redo();
        assert_eq!(tree.get_value_at([0, 0, 0]), Some(2));
        assert!(!tree.edits.can_redo());
    }

    #[test]
    fn test_get_value_at_various_positions() {
        let mut values = vec![0_u8; 16 * 16 * 16];
        // Place specific values at known positions
        values[0] = 1; // (0,0,0)
        values[1 + 2 * 16 + 3 * 16 * 16] = 5; // (1,2,3)
        values[15 + 15 * 16 + 15 * 16 * 16] = 9; // (15,15,15)

        let tree = Tree64::new((&values[..], [16, 16, 16]));

        assert_eq!(tree.get_value_at([0, 0, 0]), Some(1));
        assert_eq!(tree.get_value_at([1, 2, 3]), Some(5));
        assert_eq!(tree.get_value_at([15, 15, 15]), Some(9));
        assert_eq!(tree.get_value_at([5, 5, 5]), None);
    }

    #[test]
    fn test_expand() {
        let values = [1_u8; 4 * 4 * 4];
        let mut tree = Tree64::new((&values[..], [4; 3]));

        let old_levels = tree.root_state().num_levels;
        tree.expand();
        let new_levels = tree.root_state().num_levels;

        assert_eq!(new_levels, old_levels + 1);
    }

    #[test]
    fn test_empty_tree_construction() {
        let values = [0_u8; 4 * 4 * 4];
        let tree = Tree64::new((&values[..], [4; 3]));

        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    assert_eq!(tree.get_value_at([x, y, z]), None);
                }
            }
        }
    }

    #[test]
    fn test_full_tree_construction() {
        let values = [7_u8; 4 * 4 * 4];
        let tree = Tree64::new((&values[..], [4; 3]));

        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    assert_eq!(tree.get_value_at([x, y, z]), Some(7));
                }
            }
        }
    }
}
