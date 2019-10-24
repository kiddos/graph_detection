import { Point } from './point';

describe('Point', () => {
  it('should create an instance', () => {
    expect(new Point(0, 0)).toBeTruthy();

    expect(new Point(-100, -100)).toBeTruthy();
    expect(new Point(100, 100)).toBeTruthy();
  });
});
