import { BBox } from './bbox';
import { Point } from './point';

describe('BBox', () => {
  it('should create an instance', () => {
    let p1 = new Point(0, 0);
    let p2 = new Point(100, 100);
    expect(new BBox(p1, p2)).toBeTruthy();
  });
});
