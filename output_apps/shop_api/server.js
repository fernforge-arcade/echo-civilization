const http = require('http');
const D = require('./db.js');
const { handle } = require('./app.js');
const PORT = process.env.PORT || 3000;
const server = http.createServer(function(req, res) {
  let chunks = '';
  req.on('data', function(c){ chunks += c; });
  req.on('end', function() {
    let body = null;
    if (chunks) { try { body = JSON.parse(chunks); } catch (e) { body = chunks; } }
    const url = req.url.split('?')[0];
    if (url === '/' ) { res.writeHead(200, {'content-type':'text/html'});
      return res.end(require('fs').readFileSync(__dirname + '/public/index.html')); }
    const out = handle(req.method, url, body);
    res.writeHead(out.status, { 'content-type': 'application/json',
      'access-control-allow-origin': '*' });
    res.end(JSON.stringify(out.body));
  });
});
server.listen(PORT, function(){ console.log('listening on ' + PORT); });
