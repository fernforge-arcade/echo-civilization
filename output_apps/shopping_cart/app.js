function createApp() {
  const state = { items: [], count: 0, bill: 0, rate: 0, result: 0,
                  total: 0, input: '', filter: 'all' };
  function dispatch(action, payload) {
    switch (action) {
      case "ADD_ITEM": { state.items.push({name: payload.name, price: Number(payload.price)}); break; }
      case "REMOVE_ITEM": { state.items.splice(payload, 1); break; }
      case "CHECKOUT": { state.total = state.items.reduce(function(a, b){ return a + b.price; }, 0); break; }
      case "EMPTY": { state.items = []; break; }
      default: break;
    }
  }
  return { dispatch: dispatch, getState: function(){ return state; } };
}
if (typeof module !== 'undefined') { module.exports = { createApp: createApp }; }
