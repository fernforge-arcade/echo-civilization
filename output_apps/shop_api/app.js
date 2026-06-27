const D = require('./db.js');
const V = require('./validate.js');
const ROUTES = [
  { method: "POST", pattern: "/users", resource: "users", fields: ["name"], handler: function(D,V,req){ const m=V.missing(req.body, req.fields); if(m) return {status:400, body:{error:'missing '+m}}; const row=D.insert(req.resource, req.body); return {status:201, body:row}; } },
  { method: "GET", pattern: "/users", resource: "users", fields: ["name"], handler: function(D,V,req){ return {status:200, body:D.all(req.resource)}; } },
  { method: "GET", pattern: "/users/:id", resource: "users", fields: ["name"], handler: function(D,V,req){ const row=D.find(req.resource, req.params.id); if(!row) return {status:404, body:{error:'not found'}}; return {status:200, body:row}; } },
  { method: "PUT", pattern: "/users/:id", resource: "users", fields: ["name"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; const upd=D.update(req.resource, req.params.id, req.body); return {status:200, body:(upd||cur)}; } },
  { method: "DELETE", pattern: "/users/:id", resource: "users", fields: ["name"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; D.remove(req.resource, req.params.id); return {status:204, body:null}; } },
  { method: "POST", pattern: "/products", resource: "products", fields: ["name", "price"], handler: function(D,V,req){ const m=V.missing(req.body, req.fields); if(m) return {status:400, body:{error:'missing '+m}}; const row=D.insert(req.resource, req.body); return {status:201, body:row}; } },
  { method: "GET", pattern: "/products", resource: "products", fields: ["name", "price"], handler: function(D,V,req){ return {status:200, body:D.all(req.resource)}; } },
  { method: "GET", pattern: "/products/:id", resource: "products", fields: ["name", "price"], handler: function(D,V,req){ const row=D.find(req.resource, req.params.id); if(!row) return {status:404, body:{error:'not found'}}; return {status:200, body:row}; } },
  { method: "PUT", pattern: "/products/:id", resource: "products", fields: ["name", "price"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; const upd=D.update(req.resource, req.params.id, req.body); return {status:200, body:(upd||cur)}; } },
  { method: "DELETE", pattern: "/products/:id", resource: "products", fields: ["name", "price"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; D.remove(req.resource, req.params.id); return {status:204, body:null}; } },
  { method: "POST", pattern: "/orders", resource: "orders", fields: ["item"], handler: function(D,V,req){ const m=V.missing(req.body, req.fields); if(m) return {status:400, body:{error:'missing '+m}}; const row=D.insert(req.resource, req.body); return {status:201, body:row}; } },
  { method: "GET", pattern: "/orders", resource: "orders", fields: ["item"], handler: function(D,V,req){ return {status:200, body:D.all(req.resource)}; } },
  { method: "GET", pattern: "/orders/:id", resource: "orders", fields: ["item"], handler: function(D,V,req){ const row=D.find(req.resource, req.params.id); if(!row) return {status:404, body:{error:'not found'}}; return {status:200, body:row}; } },
  { method: "PUT", pattern: "/orders/:id", resource: "orders", fields: ["item"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; const upd=D.update(req.resource, req.params.id, req.body); return {status:200, body:(upd||cur)}; } },
  { method: "DELETE", pattern: "/orders/:id", resource: "orders", fields: ["item"], handler: function(D,V,req){ const cur=D.find(req.resource, req.params.id); if(!cur) return {status:404, body:{error:'not found'}}; D.remove(req.resource, req.params.id); return {status:204, body:null}; } }
];
function matchPath(pattern, path) {
  const ps = pattern.split('/'), xs = path.split('/');
  if (ps.length !== xs.length) return null;
  const params = {};
  for (let i = 0; i < ps.length; i++) {
    if (ps[i].charAt(0) === ':') params[ps[i].slice(1)] = xs[i];
    else if (ps[i] !== xs[i]) return null;
  }
  return params;
}
function handle(method, path, body) {
  for (const r of ROUTES) {
    if (r.method !== method) continue;
    const params = matchPath(r.pattern, path);
    if (!params) continue;
    return r.handler(D, V, { params: params, body: body, resource: r.resource, fields: r.fields });
  }
  return { status: 404, body: { error: 'no route' } };
}
module.exports = { handle: handle };
