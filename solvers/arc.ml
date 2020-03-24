open Core
open Client
open Timeout
open Utils
open Program
open Task
open Type

(* Types and Helpers*)

type block = {points : ((int*int)*int) list; original_grid : ((int*int)*int) list} ;;
type tile = {point : ((int*int)*int); block : block} ;;


let (|?) maybe default =
  match maybe with
  | Some v -> v
  | None -> Lazy.force default

let (===) block1 block2 : bool = 
  let block2_val key = (List.Assoc.find block2.points ~equal:(=) key) |? lazy (-1) in
  let block2_has_all_points = List.fold block1.points ~init:true ~f:(fun acc (key,c) -> acc && ((block2_val key) = c)) in
  (block2_has_all_points && ((List.length block1.points) = (List.length block2.points)))

let (--) i j =
  let rec from i j l =
    if i>j then l
    else from i (j-1) (j::l)
    in from i j [] ;;

let empty_grid height width color = 
let indices = List.cartesian_product (0 -- height) (0 -- width) in
let points = List.map ~f:(fun (y,x) -> ((y,x), color)) indices in
points

module IntPair = struct
  module T = struct
    type t = int * int
    let compare x y = Tuple2.compare ~cmp1:Int.compare ~cmp2:Int.compare x y
    let sexp_of_t = Tuple2.sexp_of_t Int.sexp_of_t Int.sexp_of_t
    let t_of_sexp = Tuple2.t_of_sexp Int.t_of_sexp Int.t_of_sexp
    let hash = Hashtbl.hash
  end

  include T
  include Comparable.Make(T)
end

let rec print_points = function 
[] -> printf "\n"
| ((x,y),c)::l -> printf "%d,%d:%d" x y c ; print_string " " ; print_points l

let rec print_coords = function 
[] -> printf "\n"
| (x,y)::l -> printf "%d,%d" x y ; print_string " " ; print_coords l

let rec print_list = function 
[] -> ()
| e::l -> printf "%d" e ; print_string " " ; print_list l ;;

let contains item list = 
List.mem list item ~equal:(=)

let (|?) maybe default =
  match maybe with
  | Some v -> v
  | None -> Lazy.force default

let block_of_points points original_grid = match points with
  | [] -> raise (Failure ("Empty points"))
  | points -> {points; original_grid}

(* DSL *)

let get_max_y {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum y)) points
let get_max_x {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum x)) points
let get_min_y {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum y)) points
let get_min_x {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum x)) points

let nth_highest_color block n = 
  let colors = (1 -- 9) in
  let counts = List.map colors ~f:(fun color -> (List.count block.points ~f:(fun ((y,x),c) -> c = color))) in
  let counts_with_idx = List.mapi counts ~f:(fun i x -> (i, x)) in
  let sorted = List.sort counts_with_idx ~compare:(fun (i_x, x) (i_y, y) -> (Int.descending x y)) in 
  let color,count = List.nth_exn sorted (n-1) in
  color+1 ;;

let to_original_grid_overlay {points ; original_grid} with_original = 
  let maxY = get_max_y {points=original_grid;original_grid} in
  let maxX = get_max_x {points=original_grid;original_grid} in
  let indices = List.cartesian_product (0 -- maxY) (0 -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> if with_original then match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0
      else 0 in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  block_of_points points original_grid


let print_block {points ; original_grid}  =
  printf "\n Block has %d tiles" (List.length points);
  let maxY = get_max_y {points=(original_grid @ points) ;original_grid} in
  let maxX = get_max_x {points=(original_grid @ points);original_grid} in
  let indices = List.cartesian_product (0 -- maxY) (0 -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> (-1) in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  let rec print_points points last_row = 
    match points with 
    | [] -> Printf.printf "\n\n";
    | ((y, x),c) :: rest -> if y > last_row then 
      (if (c > (-1)) then Printf.printf "\n |%i|" c else  Printf.printf "\n | |") else
      (if (c > (-1)) then Printf.printf "%i|" c else  Printf.printf " |");
      print_points rest y; in
    print_points points (-1)


let to_min_grid {points;original_grid} with_original = 
  let minY = get_min_y {points;original_grid} in
  let minX = get_min_x {points;original_grid} in
  let shiftY = (get_max_y {points;original_grid}) - minY in 
  let shiftX = (get_max_x {points;original_grid}) - minX in
  let indices = List.cartesian_product (0 -- shiftY) (0 -- shiftX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> if with_original then match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0
      else 0 in
  let new_points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y+minY,x+minX))) indices in
  block_of_points new_points original_grid;;


let print_blocks blocks = List.iter blocks ~f:(fun block -> print_block block)

let filter_blocks f l = List.filter l ~f:(fun block -> (f block))

let map_blocks f l = List.map l ~f:(fun block -> (f block))

let reflect {points;original_grid} is_horizontal = 
  let reflect_point ((y,x),c) = if is_horizontal then ((get_min_y {points;original_grid} - y + get_max_y {points;original_grid} ,x),c)
  else ((y, get_min_x {points;original_grid} - x + get_max_x {points;original_grid}),c) in
  let points = List.map ~f:reflect_point points in 
  {points;original_grid}

let move {points;original_grid} y x keep_original = 
  let new_block_points = List.map ~f:(fun ((y_pos,x_pos), color) -> ((y_pos+y, x_pos+x), color)) points in 
  (if keep_original then {points=new_block_points @ original_grid ; original_grid = original_grid} else {points=new_block_points; original_grid = original_grid})

let grow {points;original_grid} n = 
  let grow_tile_x ((y_pos,x_pos), color) = List.map ~f:(fun i -> ((y_pos,(n+1)*x_pos+i), color)) (0 -- n) in
  let nested_points = List.map ~f:grow_tile_x points in
  let temp_along_x = List.reduce nested_points ~f:(fun a b -> a @ b) in 
  let final_points = match temp_along_x with 
  | None -> []
  | Some(temp_along_x) -> 
    let grow_tile_y ((y_pos,x_pos), color) = List.map ~f:(fun i -> (((n+1)*y_pos+i,x_pos), color)) (0 -- n) in
    let along_y_and_x = List.map ~f:grow_tile_y temp_along_x in
    (List.reduce along_y_and_x ~f:(fun a b -> a @ b) |? lazy []) in
  {points = final_points ; original_grid}
  
let merge a b =
  let rec add_until_empty list1 list2 = 
    match list1 with
    | [] -> list2
    | el :: rest -> add_until_empty rest (el :: list2) in
  let points = add_until_empty a.points b.points in
  let original_grid = a.original_grid in
  {points ; original_grid}

(* let duplicate block tdirection = 
  match tdirection with 
  | "right" -> let x_shift = (get_max_x block) - (get_min_x block) + 1 in 
  move block 0 x_shift true
  | "down" -> let y_shift = (get_max_y block) - (get_min_y block) + 1 in  
  move block y_shift 0 true 
  | _ -> block ;; *)

let is_rectangle block full = 
  (* TODO: Implement non-full version *)
  let {points;original_grid} = to_min_grid block false in
  (List.length points) = (List.length block.points)

let split block is_horizontal =
  let {points ; original_grid} = block in
  match is_horizontal with 
  | true ->
  let horizontal_length = (get_max_y block) - (get_min_y block) + 1 in
  let halfway = (get_min_y block) + (horizontal_length / 2) in 
  let top_half = List.filter points ~f: (fun ((y,x),c) -> y < halfway) in
  let bottom_half_start = if ((horizontal_length mod 2) = 1) then halfway + 1 else halfway in
  let bottom_half = List.filter points ~f: (fun ((y,x),c) -> y >= bottom_half_start) in
  [{points = top_half; original_grid = original_grid}; {points = bottom_half; original_grid = original_grid}]
  | false ->
  let vertical_length = (get_max_x block) - (get_min_x block) + 1 in
  let halfway = (get_min_x block) + (vertical_length / 2) in 
  let left_half = List.filter points ~f: (fun ((y,x),c) -> x < halfway) in
  let right_half_start = if ((vertical_length mod 2) = 1) then halfway + 1 else halfway in
  let right_half = List.filter points ~f: (fun ((y,x),c) -> x >= right_half_start) in
  [{points = left_half; original_grid = original_grid}; {points = right_half; original_grid = original_grid}]

let is_symmetrical block is_horizontal = 
  let split_block = split block is_horizontal in
  let reflected_split_block = split (reflect block is_horizontal) is_horizontal in
  match split_block with
  | [] -> false
  | first_half :: _ -> 
  match reflected_split_block with 
    | first_reflected_half :: _ -> first_half === first_reflected_half
    | _ -> false
  ;;

let has_min_tiles block n = List.length block.points >= n ;;

let box_block {points;original_grid} = 
  let minY = get_min_y {points;original_grid} in
  let maxY = get_max_y {points;original_grid} in
  let minX = get_min_x {points;original_grid} in
  let maxX = get_max_x {points;original_grid} in
  let indices = List.cartesian_product (minY -- maxY) (minX -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0 in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
   {points ; original_grid}

let to_int_pair x = (IntPair.Set.choose (IntPair.Set.singleton x)) |? lazy (raise (Failure ("x is empty"))) ;;
let to_tuple (a,b) = (a,b) ;;


let create_edge_map block colors use_corners = 
  let vertex_points = List.filter block.points ~f:(fun ((y,x),c) -> contains c colors) in
  let vertices = List.map vertex_points ~f:(fun ((y,x),c) -> (y,x)) in
  (* only include downward edges to avoid cycles *) 
  let basic_adjacent = [(1,0);(0,1);(0,-1);(-1,0)] in
  let adjacent = if use_corners then (basic_adjacent @ [(1,1);(1,-1);(-1,-1);(-1,1)]) else basic_adjacent in
  let vertex_edges (y,x) = List.filter adjacent ~f:(fun (y_inc,x_inc) -> contains (y+y_inc,x+x_inc) vertices) in
  let edge_map = List.map vertices ~f:(fun (y,x) -> ((y,x),(List.map (vertex_edges (y,x)) ~f:(fun (y_edge, x_edge) -> y_edge+y, x_edge+x)))) in
  (* let vertices = List.map (List.filter edge_map ~f:(fun (v,e) -> List.length e > 0)) ~f:(fun (v,e) -> v) in *)
  (* List.iter edge_map ~f:(fun (key,vals) -> print_coords vals); *)
  (vertices, edge_map) ;;

let dfs graph visited start_node = 
  let rec explore path visited node = 
    if (List.mem path node ~equal:(=)) then visited else     
      let new_path = node :: path in 
      let edges    = (List.Assoc.find graph node ~equal:(=)) |? lazy [] in
      let new_edges = List.filter edges ~f:(fun e -> not (List.mem visited e ~equal:(=))) in
      let visited  = List.fold_left ~f:(explore new_path) ~init:visited new_edges in
      node :: visited
  in explore [] visited start_node

let find_blocks_by block colors is_corner box_blocks = 
  let vertices, graph = create_edge_map block colors is_corner in 
  
  let state_to_set state = List.fold_left state ~init:IntPair.Set.empty ~f:IntPair.Set.union in 

  let explore_v state v = 
    (* if IntPair.Set.mem (List.fold_left state ~init:IntPair.Set.empty ~f:IntPair.Set.union) (to_int_pair v |? lazy (raise (Failure ("No tile")))) *)
    if IntPair.Set.mem (state_to_set state) (to_int_pair v)
      then state
    else 
      let connected_component_vertices = IntPair.Set.of_list (List.map (dfs graph [] v) ~f:to_int_pair) in
      state @ [connected_component_vertices] in

  let init = [IntPair.Set.empty] in
  let connected_components = List.fold_left ~f:explore_v ~init:init (List.map vertices ~f:to_int_pair) in
  let connected_components_2d_list = List.map connected_components ~f:(fun cc -> 
    let list_cc = IntPair.Set.to_list cc in
    List.map list_cc ~f:(fun int_pair -> to_tuple int_pair)) in
  let deduce_val key = match (List.Assoc.find block.points key ~equal:(=)) with 
  | None -> 0
  | Some c -> c in
  let with_empty = List.map connected_components_2d_list ~f:(fun cc -> 
    {points = List.map cc ~f:(fun key -> (key, deduce_val key)); original_grid = block.original_grid}) in 
  let final_blocks = List.drop with_empty 1 in
  if box_blocks then List.map final_blocks ~f:box_block else final_blocks ;;

let find_blocks_by_color block color is_corner box_blocks = 
  find_blocks_by block [color] is_corner box_blocks ;;

let find_blocks_by_black_b block is_corner box_blocks = 
  find_blocks_by block (1 -- 9) is_corner box_blocks

let find_same_color_blocks block is_corner box_blocks = 
  let blocks_by_color = List.map (1 -- 9) ~f:(fun color -> find_blocks_by block [color] is_corner box_blocks) in
  List.concat blocks_by_color ;;


let fill_color block new_color = 
  let points = List.map block.points ~f:(fun ((y,x),_) -> (y,x),new_color) in
  block_of_points points block.original_grid

let replace_color block old_color new_color = 
  let points = List.map block.points ~f:(fun ((y,x),c) -> (y,x), if (c = old_color) then new_color else c) in
  {points = points ; original_grid = block.original_grid} ;;

let merge_blocks blocks = 
  match blocks with
  | [] -> raise (Failure ("Merge with empty list"))
  | l -> let merged = (List.reduce l ~f:merge) in match merged with
    | None -> raise (Failure ("Merge with empty list"))
    | Some(block) -> block

let filter_tiles block f = 
  let new_points = List.filter block.points ~f:(fun point -> f {point ; block = block}) in
  block_of_points new_points block.original_grid

let is_interior tile is_corner = 
  let adjacent = [(1,0);(0,1);(0,-1);(-1,0)] in
  let corner_adjacent = [(1,1);(1,-1);(-1,-1);(-1,1)] in
  let neighbors = if is_corner then (adjacent @ corner_adjacent) else adjacent in
  let ((y_tile, x_tile), c_tile) = tile.point in
  let actual_neighbors = List.filter_map neighbors ~f:(fun (y,x) -> List.Assoc.find tile.block.points ~equal:(=) (y+y_tile,x+x_tile)) in
  let expected_num_neighbors = if is_corner then 8 else 4 in 
((List.length actual_neighbors) = expected_num_neighbors) ;;

let nth_of_sorted_object_list objects f n = 
  let with_ints = List.map objects ~f:(fun x -> (f x, x)) in
  let sorted_with_ints = List.sort with_ints ~compare:(fun (a_v, a_x) (b_v, b_x) -> b_v - a_v) in
  let sorted = List.map sorted_with_ints ~f:(fun (v, x) -> x) in
  List.nth_exn sorted n ;;

let remove_black_b block = 
  let new_points = List.filter block.points ~f:(fun ((y,x),c) -> not (c = 0)) in
  block_of_points new_points block.original_grid ;; 

let nth_primary_color block n = 
  let get_color_count block color = 
      List.fold_left ~init:0 ~f:(fun count ((y,x),c) -> if (c = color) then (count + 1) else count) block.points in
  let color_counts = List.map ~f:(fun color -> (color, get_color_count block color)) (0 -- 9) in 
  let sorted_colors_with_ints = List.sort color_counts ~compare:(fun (a_color, a_count) (b_color, b_count) -> b_count - a_count) in
  let nth_color, count = List.nth_exn sorted_colors_with_ints n in 
  nth_color ;;


let overlap_split_blocks split_blocks f_tile = 
  let center_blocks = List.map split_blocks ~f:(fun block -> to_min_grid block false) in

  let overlap_tilewise a b f_tile = 
    let overlapped = List.map a.points ~f:(fun ((a_y,a_x),a_c) -> 
      let b_c = List.Assoc.find_exn b.points ~equal:(=) (a_y,a_x) in
      ((a_y,a_x), (f_tile a_c b_c))) in 
    block_of_points overlapped a.original_grid in

  List.reduce_exn center_blocks ~f:(fun state el -> overlap_tilewise state el f_tile);;

let color_logical c_1 c_2 new_color binary_f = 
  let binary_1 = if c_1 > 0 then 1 else 0 in
  let binary_2 = if c_2 > 0 then 1 else 0 in
  let flag = (binary_f binary_1 binary_2) in
  if (flag = 1) then new_color else 0 ;;



register_special_task "arc" (fun extras ?timeout:(timeout = 0.001) name ty examples ->
(* Printf.eprintf "Making an arc task %s \n" name; *)
{ name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun p -> 
        (* Printf.eprintf "Program: %s \n" (string_of_program p) ; *)
        (* flush_everything () ; *)
        let p = analyze_lazy_evaluation p in
        let rec loop = function
          | [] -> true
          | (xs,y) :: e ->
            try
              match run_for_interval
                      timeout
                      (fun () -> (magical (run_lazy_analyzed_with_arguments p xs)) === (magical y))
              with
                | Some(true) -> loop e
                | _ -> false
            with (* We have to be a bit careful with exceptions if the
                  * synthesized program generated an exception, then we just
                  * terminate w/ false but if the enumeration timeout was
                  * triggered during program evaluation, we need to pass the
                  * exception on
                  *)
              | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
              | EnumerationTimeout  -> raise EnumerationTimeout
              | _                   -> false
        in
        if loop examples
          then 0.0
          else log 0.0)
}) ;;


(* primitives *)

ignore(primitive "black" tcolor 0) ;;
ignore(primitive "blue" tcolor 1) ;;
ignore(primitive "red" tcolor 2) ;;
ignore(primitive "green" tcolor 3) ;;
ignore(primitive "yellow" tcolor 4) ;;
ignore(primitive "grey" tcolor 5) ;;
ignore(primitive "pink" tcolor 6) ;;
ignore(primitive "orange" tcolor 7) ;;
ignore(primitive "teal" tcolor 8) ;;
ignore(primitive "maroon" tcolor 9) ;;

(* tblocks -> tblock *)
ignore(primitive "merge_blocks" (tblocks @> tblock) merge_blocks) ;;
ignore(primitive "filter_blocks" ((tblock @> tboolean) @> tblocks @> tblocks) (fun f l -> List.filter ~f:f l));;
ignore(primitive "map_blocks" ((tblock @> tblock) @> tblocks @> tblocks) (fun f l -> List.map ~f:f l));;
ignore(primitive "nth_of_sorted_object_list" (tblocks @> (tblock @> tint) @> tint @> tblock) nth_of_sorted_object_list) ;;

(* tblock -> tblock *)
ignore(primitive "reflect" (tblock @> tboolean @> tblock) reflect) ;;
ignore(primitive "move" (tblock @> tint @> tint @> tboolean @> tblock) move) ;;
ignore(primitive "grow" (tblock @> tint @> tblock) grow) ;;
ignore(primitive "fill_color" (tblock @> tcolor @> tblock) fill_color) ;;
ignore(primitive "replace_color" (tblock @> tcolor @> tcolor @> tblock) replace_color) ;;
ignore(primitive "remove_black_b" (tblock @> tblock) remove_black_b) ;;
ignore(primitive "box_block" (tblock @> tblock) box_block) ;;
ignore(primitive "filter_tiles" (tblock @> (ttile @> tboolean) @> tblock) filter_tiles) ;;
(* tblock -> tgridout *)
ignore(primitive "to_min_grid" (tblock @> tboolean @> tgridout) to_min_grid) ;;
ignore(primitive "to_original_grid_overlay" (tblock @> tboolean @> tgridout) to_original_grid_overlay) ;;
(* tblock -> tint *)
ignore(primitive "get_height" (tblock @> tint) (fun block -> (get_max_y block) - (get_min_y block))) ;;
ignore(primitive "get_width" (tblock @> tint) (fun block -> (get_max_x block) - (get_max_x block))) ;;
ignore(primitive "get_num_tiles" (tblock @> tint) (fun {points;original_grid} -> List.length points)) ;;
(* tblock -> tcolor *)
ignore(primitive "nth_primary_color" (tblock @> tint @> tcolor) nth_primary_color) ;;
(* tblock -> tboolean *)
ignore(primitive "is_symmetrical" (tblock @> tboolean @> tboolean) is_symmetrical) ;;
ignore(primitive "is_rectangle" (tblock @> tboolean @> tboolean) is_rectangle) ;;
ignore(primitive "has_min_tiles" (tblock @> tint @> tboolean) has_min_tiles) ;;

(* tgridin -> tblocks *)
ignore(primitive "grid_to_block" (tgridin @> tblock) (fun x -> x)) ;;
ignore(primitive "find_same_color_blocks" (tgridin @> tboolean @> tboolean @> tblocks) find_same_color_blocks) ;;
ignore(primitive "find_blocks_by_black_b" (tgridin @> tboolean @> tboolean @> tblocks) find_blocks_by_black_b) ;;
ignore(primitive "find_blocks_by_color" (tgridin @> tcolor @> tboolean @> tboolean @> tblocks) find_blocks_by_color) ;;
(* tgridin -> tsplitblocks *)
ignore(primitive "split_grid" (tgridin @> tboolean @> tsplitblocks) split) ;;

(* ttile -> tboolean *)
ignore(primitive "is_interior" (ttile @> tboolean @> tboolean) is_interior) ;;
(* ttile -> tblock *)
ignore(primitive "to_block" (ttile @> tblock) (fun tile -> {points=[tile.point] ; original_grid = tile.block.original_grid})) ;;

(* tsplitblocks -> tgridout *)
ignore(primitive "overlap_split_blocks" (tsplitblocks @> (tcolor @> tcolor @> tcolor) @> tgridout) overlap_split_blocks) ;;
ignore(primitive "to_blocks" (tsplitblocks @> tblocks) (fun blocks -> blocks)) ;;


(* tcolor -> tcolor *)
ignore(primitive "color_logical" (tcolor @> tcolor @> tcolor @> tlogical @> tcolor) color_logical) ;;

(* tlogical *)
ignore(primitive "land" tlogical (land)) ;;
ignore(primitive "lor" tlogical (lor)) ;;
ignore(primitive "lxor" tlogical (lxor)) ;;


















let python_split x =
  let split = String.split_on_chars ~on:[','] x in 
  let filt_split = List.filter split ~f:(fun x -> x <> "") in
  let y = List.nth_exn filt_split 0 |> int_of_string in
  let x = List.nth_exn filt_split 1 |> int_of_string in
  (y,x)
;;

let to_grid task = 
  let open Yojson.Basic.Util in
  let json = task |> member "grid" |> to_assoc in 
  let grid_points = List.map json ~f:(fun (key, color) -> ((python_split key), (to_int color))) in
  let grid = {points = grid_points ; original_grid = grid_points} in 
  (* print_block grid; *)
  grid ;;

let convert_raw_to_block raw = 
  let open Yojson.Basic.Util in
  let y_length = List.length (raw |> to_list) -1 in
  let x_length = List.length (List.nth_exn (raw |> to_list) 0 |> to_list) - 1 in
  let indices = List.cartesian_product (0 -- y_length) (0 -- x_length) in
  let match_row row x = match List.nth row x with
        | Some c -> c |> to_int
        | None -> (-1) in
  let deduce_val (y,x) = match (List.nth (raw |> to_list) y) with
      | Some row -> match_row (to_list row) x 
      | None -> (-1) in
  let new_points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  {points = new_points; original_grid = new_points} ;;


let test_example assoc_list p = 
  let open Yojson.Basic.Util in
  let raw_input = List.Assoc.find_exn assoc_list "input" ~equal:(=) in
  let raw_expected_output = List.Assoc.find_exn assoc_list "output" ~equal:(=) in
  let input = convert_raw_to_block raw_input in
  let expected_output = convert_raw_to_block raw_expected_output in
  let got_output = p input in
  let matched = got_output === expected_output in
  printf "\n%B\n" matched;
  match matched with 
  | false -> 
    printf "\n Input \n";
    print_block input ;
    printf "\n Resulting Output \n";
    print_block got_output;
    printf "\n Expected Output \n";
    print_block expected_output;
  | true -> ();;

let test_task file_name ex p =
  printf "\n ----------------------------- Task: %s --------------------------- \n" file_name;
  let fullpath = String.concat ["/Users/theo/Development/program_induction/ec/arc-data/data/training/"; file_name; ".json"] in
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let json = from_file fullpath in
  let json = json |> member "train" |> to_list in
  let pair_list = List.map json ~f:(fun pair -> pair |> to_assoc) in 
  match ex with
  | 0 -> test_example (List.nth_exn pair_list 0) p
  | 1 -> test_example (List.nth_exn pair_list 1) p
  | 2 -> test_example (List.nth_exn pair_list 2) p
  | _ -> List.iter pair_list ~f:(fun assoc_list -> test_example assoc_list p) ;;


let p_72ca375d grid = 
  let blocks = find_same_color_blocks grid true false in
  let filtered_blocks = List.filter blocks ~f:(fun block -> is_symmetrical block false) in
  let merged_block = merge_blocks filtered_blocks in
  to_min_grid merged_block false ;;
(* test_task "72ca375d" (-1) p_72ca375d ;; *)


let p_5521c0d9 grid = 
  let blocks = find_same_color_blocks grid true false in
  let get_height block = ((get_max_y block) - (get_min_y block)) + 1 in
  let shifted_blocks = map_blocks (fun block -> move block (-(get_height block)) 0 false) blocks in
  let merged_blocks = merge_blocks shifted_blocks in
  to_original_grid_overlay merged_blocks false ;;
(* test_task "5521c0d9" (-1) p_5521c0d9;; *)

let p_f25fbde4 grid = 
  let blocks = find_blocks_by_black_b grid true false in 
  let block = merge_blocks blocks in
  let grow_block = grow block 1 in
  to_min_grid grow_block false ;;
(* test_task "f25fbde4" (-1) p_f25fbde4;; *)

let p_50cb2852 grid = 
  let blocks = find_blocks_by_black_b grid true false  in
  let interior_blocks = map_blocks (fun block -> filter_tiles block (fun tile -> is_interior tile true)) blocks in
  let filled_interior_blocks = map_blocks (fun block -> fill_color block 8) interior_blocks in
  let merged_blocks = merge_blocks filled_interior_blocks in
  to_original_grid_overlay merged_blocks true ;;
(* test_task "50cb2852" (-1) p_50cb2852 ;; *)


let p_fcb5c309 grid = 
  let blocks = find_same_color_blocks grid true true in
  let largest_block = nth_of_sorted_object_list blocks (fun block -> List.length block.points) 0 in
  let largest_block_no_b = remove_black_b largest_block in
  let colored_block = replace_color largest_block (nth_primary_color largest_block_no_b 0) (nth_primary_color largest_block_no_b 1) in
  to_min_grid colored_block false ;;
(* test_task "fcb5c309" (-1) p_fcb5c309 ;; *)

let p_ce4f8723 grid = 
  let split_blocks = split grid true in
  overlap_split_blocks split_blocks (fun a b -> color_logical a b 3 (lor)) ;;
(* test_task "ce4f8723" (-1) p_ce4f8723 ;; *)

let p_0520fde7 grid = 
  let split_blocks = split grid false in
  overlap_split_blocks split_blocks (fun a b -> color_logical a b 2 (land)) ;;
(* test_task "0520fde7" (-1) p_0520fde7 ;; *)

(* 
let example_grid = {points = [((1,3),4); ((1,2),4); ((1,1),4); ((1,4),4); ((2,4),4); ((3,4),4); ((4,4),3); ((2,3),4); ((2,2),4); ((2,1),4); ((3,3),4); ((3,2),4); ((3,1),4); ((4,3),4); ((4,2),4); ((4,1),4)] ; original_grid = empty_grid 4 4 0} in
let blocks = find_blocks_by_color example_grid 4 false false in 
let block = List.nth_exn blocks 0 in
print_block block;
let filtered_block = filter_tiles block (fun tile -> is_interior true block) in
print_block filtered_block ;;


 *)