// validation layer — returns the name of the first missing required field, or null
function missing(body, fields) {
  for (const f of (fields || [])) {
    if (body == null || body[f] === undefined || body[f] === null || body[f] === '') return f;
  }
  return null;
}
module.exports = { missing: missing };
