// data layer — in-memory collections with auto-increment ids
let store = {};
function reset() { store = {}; }
function coll(name) { if (!store[name]) store[name] = []; return store[name]; }
function insert(name, rec) {
  const c = coll(name);
  const id = c.length ? Math.max.apply(null, c.map(function(r){return r.id;})) + 1 : 1;
  const row = Object.assign({ id: id }, rec || {});
  c.push(row); return row;
}
function all(name) { return coll(name).slice(); }
function find(name, id) { return coll(name).find(function(r){ return r.id === Number(id); }); }
function update(name, id, patch) {
  const r = find(name, id); if (!r) return null; Object.assign(r, patch || {}); return r;
}
function remove(name, id) {
  const c = coll(name); const i = c.findIndex(function(r){ return r.id === Number(id); });
  if (i < 0) return false; c.splice(i, 1); return true;
}
module.exports = { reset: reset, insert: insert, all: all, find: find, update: update, remove: remove };
